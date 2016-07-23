#include <numeric>
#include <iostream>

#include <moderngpu/transform.hxx>   // for cta_launch.
#include <moderngpu/memory.hxx>      // for mem_t.
#include <moderngpu/kernel_scan.hxx>
#include "measure_bandwidth_of_invocation.hpp"

using namespace mgpu;


template<size_t group_size, size_t grain_size, typename input_it,  typename output_it, typename op_t>
void inclusive_scan_tiles(input_it input, int count, output_it output, op_t op, context_t& context)
{
  constexpr size_t tile_size = group_size * grain_size;

  using launch_t = launch_box_t<
    arch_52_cta<group_size, grain_size>
  >;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);

  auto downsweep_k = [=] __device__ (int tid, int cta)
  {
    typedef cta_scan_t<group_size, type_t> scan_t;

    __shared__ union {
      typename scan_t::storage_t scan;
      type_t values[tile_size];
    } shared;

    // Load a tile to register in thread order.
    range_t tile = get_tile(cta, tile_size, count);
    array_t<type_t, grain_size> x = mem_to_reg_thread<group_size, grain_size>(input + tile.begin, tid, tile.count(), shared.values);

    // Scan the array with carry-in from the partials.
    array_t<type_t, grain_size> y = scan_t().scan(tid, x, shared.scan, 0, cta > 0, tile.count(), op, type_t(), mgpu::scan_type_inc).scan;

    // Store the scanned values to the output.
    reg_to_mem_thread<group_size, grain_size>(y, tid, tile.count(), output + tile.begin, shared.values);    
  };
  cta_transform<launch_t>(downsweep_k, count, context);
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

  std::vector<int> input_host(n, 1);

  // Copy the data to the GPU.
  mem_t<int> input_device = to_mem(input_host, context);

  mem_t<int> output_device(n, context);
  inclusive_scan_tiles<group_size, grain_size>(input_device.data(), input_device.size(), output_device.data(), plus_t<int>(), context);

  // Get the result.
  std::vector<int> output_host = from_mem(output_device);

  // compare to reference
  std::vector<int> reference(n);

  for(size_t i = 0; i < n; i += tile_size)
  {
    auto tile_begin = reference.begin() + i;
    auto tile_end = reference.begin() + min(i + tile_size, reference.size());
    std::iota(tile_begin, tile_end, 1);
  }

  assert(reference == output_host);

  auto bandwidth = measure_bandwidth_of_invocation_in_gigabytes_per_second(100, 2 * sizeof(int) * n, [&]
  {
    inclusive_scan_tiles<group_size,grain_size>(input_device.data(), input_device.size(), output_device.data(), plus_t<int>(), context);
    context.synchronize();
  });

  std::cout << "N: " << n << std::endl;
  std::cout << "Mean bandwidth: " << bandwidth << " GB/s " << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

