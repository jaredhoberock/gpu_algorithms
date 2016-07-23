#include <numeric>
#include <iostream>

#include <moderngpu/transform.hxx>   // for cta_launch.
#include <moderngpu/memory.hxx>      // for mem_t.
#include <moderngpu/kernel_scan.hxx>
#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <agency/experimental.hpp>
#include "measure_bandwidth_of_invocation.hpp"


template<size_t block_size, size_t grain_size = 1>
auto static_grid(int num_blocks) ->
  decltype(agency::cuda::par(num_blocks, agency::cuda::experimental::static_concurrent_execution_policy<block_size, grain_size>()))
{
  return agency::cuda::par(num_blocks, agency::cuda::experimental::static_concurrent_execution_policy<block_size, grain_size>());
}

template<size_t block_size, size_t grain_size = 1>
using static_grid_agent = agency::parallel_group<agency::experimental::static_concurrent_agent<block_size, grain_size>>;

using namespace mgpu;


template<size_t group_size, size_t grain_size, typename input_it1, typename input_it2,  typename output_it, typename op_t>
void inclusive_scan_tiles(input_it1 input, int count, input_it2 inits, output_it output, op_t op, context_t& context)
{
  constexpr size_t tile_size = group_size * grain_size;

  using launch_t = launch_box_t<
    arch_52_cta<group_size, grain_size>
  >;

  typedef typename std::iterator_traits<input_it1>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);

  auto downsweep_k = [=] __device__ (static_grid_agent<group_size,grain_size>& self)
  {
    int tid = self.inner().index();
    int cta = self.outer().index();

    typedef cta_scan_t<group_size, type_t> scan_t;

    __shared__ union {
      typename scan_t::storage_t scan;
      type_t values[tile_size];
    } shared;

    // Load a tile to register in thread order.
    range_t tile = get_tile(cta, tile_size, count);
    array_t<type_t, grain_size> x = mem_to_reg_thread<group_size, grain_size>(input + tile.begin, tid, tile.count(), shared.values);

    // Scan the array with carry-in from the partials.
    array_t<type_t, grain_size> y = scan_t().scan(tid, x, shared.scan, inits[cta], true, tile.count(), op, type_t(), mgpu::scan_type_inc).scan;

    // Store the scanned values to the output.
    reg_to_mem_thread<group_size, grain_size>(y, tid, tile.count(), output + tile.begin, shared.values);    
  };
  agency::bulk_invoke(static_grid<group_size,grain_size>(num_ctas), downsweep_k);
}


int main(int argc, char** argv)
{
  standard_context_t context;

  size_t num_tiles = 1 << 20;

  if(argc > 1)
  {
    num_tiles = std::atoi(argv[1]);
  }

  constexpr size_t group_size = 128;
  constexpr size_t grain_size = 11;
  constexpr size_t tile_size = group_size * grain_size;
  size_t n = tile_size * num_tiles;

  std::vector<int> input_host(n);
  std::default_random_engine rng(n);
  std::generate(input_host.begin(), input_host.end(), rng);

  std::vector<int> inits_host(num_tiles);
  std::generate(inits_host.begin(), inits_host.end(), rng);

  // Copy the data to the GPU.
  mem_t<int> input_device = to_mem(input_host, context);
  mem_t<int> inits_device = to_mem(inits_host, context);

  mem_t<int> output_device(n, context);
  inclusive_scan_tiles<group_size, grain_size>(input_device.data(), input_device.size(), inits_device.data(), output_device.data(), plus_t<int>(), context);

  // Get the result.
  std::vector<int> output_host = from_mem(output_device);

  // compare to reference
  std::vector<int> reference(n);

  auto input_tiles = agency::experimental::chunk(input_host, tile_size);
  auto output_tiles = agency::experimental::chunk(reference, tile_size);

  for(int i = 0; i < input_tiles.size(); ++i)
  {
    std::partial_sum(input_tiles[i].begin(), input_tiles[i].end(), output_tiles[i].begin());
    std::for_each(output_tiles[i].begin(), output_tiles[i].end(), [&](int &x)
    {
      x += inits_host[i];
    });
  }

  assert(reference == output_host);

  auto bandwidth = measure_bandwidth_of_invocation_in_gigabytes_per_second(100, 2 * sizeof(int) * n, [&]
  {
    inclusive_scan_tiles<group_size,grain_size>(input_device.data(), input_device.size(), inits_device.data(), output_device.data(), plus_t<int>(), context);
  });

  std::cout << "N: " << n << std::endl;
  std::cout << "Mean bandwidth: " << bandwidth << " GB/s " << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

