#include <numeric>
#include <iostream>

#include <moderngpu/transform.hxx>   // for cta_launch.
#include <moderngpu/memory.hxx>      // for mem_t.
#include <moderngpu/kernel_scan.hxx>
#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <agency/experimental.hpp>
#include "measure_bandwidth_of_invocation.hpp"
#include "algorithm/copy.hpp"
#include "algorithm/for_loop.hpp"
#include "algorithm/transform.hpp"
#include "algorithm/inclusive_scan.hpp"
#include "collective_scanner.hpp"


template<size_t block_size, size_t grain_size = 1>
auto static_grid(int num_blocks) ->
  decltype(agency::cuda::par(num_blocks, agency::cuda::experimental::static_concurrent_execution_policy<block_size, grain_size>()))
{
  return agency::cuda::par(num_blocks, agency::cuda::experimental::static_concurrent_execution_policy<block_size, grain_size>());
}

template<size_t block_size, size_t grain_size = 1>
using static_grid_agent = agency::parallel_group<agency::experimental::static_concurrent_agent<block_size, grain_size>>;

using namespace mgpu;
using namespace agency::experimental;


// XXX this costs a ton of registers
template<bool exclusive, size_t group_size, size_t grain_size, class Range1, class Range2, class T, class BinaryOperation>
__device__
T collective_scan_with_carry(agency::experimental::static_concurrent_agent<group_size,grain_size>& self, const Range1& input, Range2&& output, T carry_in, BinaryOperation binary_op)
{
  using namespace agency::experimental;

  constexpr int tile_size = group_size * grain_size;

  static_assert(!exclusive, "exclusive scan unimplemented!");

  __shared__ union
  {
    collective_scanner<T,group_size> scanner;
    array<T,tile_size> tile_storage;
  } shared;

  for(int offset = 0; offset < input.size(); offset += tile_size)
  {
    auto partial_tile_size = min(tile_size, int(input.size() - offset));
    auto input_tile = counted(input, offset, bounded_int<tile_size>(partial_tile_size));

    // create a view of shared memory with as many elements as the input tile
    auto view_of_shared_tile = counted(shared.tile_storage, input_tile.size());

    // copy the input tile into shared memory
    collective_copy(self, input_tile, view_of_shared_tile);

    // tile shared memory into subtiles of size grain_size
    auto shared_subtiles = chunk(view_of_shared_tile, grain_size);
    auto view_of_shared_subtile = shared_subtiles.chunk_or_empty(self.rank());
    int num_subtiles = shared_subtiles.size();

    // copy the subtile from shared memory into registers
    short_vector<T,grain_size> local_subtile = view_of_shared_subtile;

    // each agent does an in-place inclusive scan of its local array
    ::inclusive_scan(local_subtile, local_subtile, binary_op);

    // each thread contributes a summand for a group-wide exclusive scan 
    optional<T> summand = local_subtile.back_or_none();

    // wait until all agents have their summand before using the scanner
    self.wait();

    // collectively compute the exclusive scan of the summands in-place
    carry_in = shared.scanner.inplace_exclusive_scan(self, summand, num_subtiles, carry_in, binary_op);

    // to produce the final inclusive scan, add the thread's carry-in to the scan of its local subtile
    // store each thread's result directly to the tile in shared memory
    transform(local_subtile, view_of_shared_subtile, [=](const T& element)
    {
      return binary_op(*summand, element);
    });

    // wait for each thread to complete their result
    self.wait();

    // copy the tile from shared memory into the output tile
    auto output_tile = counted(output, offset, bounded_int<tile_size>(partial_tile_size));
    collective_copy(self, view_of_shared_tile, output_tile);
  }

  return carry_in;
}


// XXX abstracting this into a function introduces a register
template<bool exclusive, size_t group_size, size_t grain_size, class Range1, class Range2, class T, class BinaryOperation>
__device__
T bounded_scan_tile(agency::experimental::static_concurrent_agent<group_size,grain_size>& self, const Range1& input_tile, Range2&& output_tile, T carry_in, BinaryOperation binary_op)
{
  using namespace agency::experimental;

  static_assert(!exclusive, "exclusive scan unimplemented!");

  __shared__ union
  {
    collective_scanner<T,group_size> scanner;
    array<T,group_size * grain_size> tile_storage;
  } shared;

  // create a view of shared memory with as many elements as the input tile
  auto view_of_shared_tile = counted(shared.tile_storage, input_tile.size());

  // copy the input tile into shared memory
  collective_copy(self, input_tile, view_of_shared_tile);

  // tile shared memory into subtiles of size grain_size
  auto shared_subtiles = chunk(view_of_shared_tile, grain_size);
  auto view_of_shared_subtile = shared_subtiles.chunk_or_empty(self.rank());
  int num_subtiles = shared_subtiles.size();

  // copy the subtile from shared memory into registers
  // XXX should call something like to_container() here
  //     statically-sized ranges would be copied into an array<T,N>
  //     bounded-sized ranges would be copied into a short_vector<T,N>
  short_vector<T,grain_size> local_subtile = view_of_shared_subtile;

  // each agent does an in-place inclusive scan of its local array
  ::inclusive_scan(local_subtile, local_subtile, binary_op);

  // each thread contributes a summand for a group-wide exclusive scan 
  optional<T> summand = local_subtile.back_or_none();

  // wait until all agents have their summand before using the scanner
  self.wait();

  // collectively compute the exclusive scan of the summands in-place
  T carry_out = shared.scanner.inplace_exclusive_scan(self, summand, num_subtiles, carry_in, binary_op);

  // to produce the final inclusive scan, add the thread's carry-in to the scan of its local subtile
  // store each thread's result directly to the tile in shared memory
  // XXX for some reason capturing by reference significantly inflates the register requirements
  //     so capture summand by value
  transform(local_subtile, view_of_shared_subtile, [=](const T& element)
  {
    return binary_op(*summand, element);
  });

  // wait for each thread to complete their result
  self.wait();

  // copy the tile from shared memory into the output tile
  collective_copy(self, view_of_shared_tile, output_tile);

  return carry_out;
}


template<size_t group_size, size_t grain_size, typename input_it1, typename input_it2, typename output_it, typename op_t>
void inclusive_scan_tiles(input_it1 input, int count, input_it2 inits, output_it output, op_t binary_op, context_t& context)
{
  constexpr int tile_size = group_size * grain_size;

  // embed tile_size into the type system
  constexpr auto bounded_tile_size = bounded_int<tile_size>(tile_size);

  typedef typename std::iterator_traits<input_it1>::value_type T;

  auto input_view  = span<T>(input, count);
  auto output_view = span<T>(output, count);

  auto input_tiles = chunk(input_view, bounded_tile_size);
  auto output_tiles = chunk(output_view, bounded_tile_size);

  auto downsweep_k = [=] __device__ (static_grid_agent<group_size,grain_size>& self)
  {
    int cta = self.outer().index();

    auto input_tile = input_tiles[cta]; 
    auto output_tile = output_tiles[cta];

    bounded_scan_tile<false>(self.inner(), input_tile, output_tile, inits[cta], binary_op); 
    //collective_scan_with_carry<false>(self.inner(), input_tile, output_tile, inits[cta], binary_op);
  };
  agency::bulk_invoke(static_grid<group_size,grain_size>(input_tiles.size()), downsweep_k);
}


int main()
{
  standard_context_t context;

  constexpr size_t group_size = 128;
  constexpr size_t grain_size = 11;
  constexpr size_t tile_size = group_size * grain_size;

  for(int i = 0; i < 30; ++i)
  {
    size_t n = 1 << i;

    std::cout << "testing n: " << n << std::endl;

    size_t num_tiles = (n + tile_size - 1) / tile_size;

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
  }

  size_t n = 1 << 30;
  size_t num_tiles = (n + tile_size - 1) / tile_size;

  std::vector<int> input_host(n);
  std::default_random_engine rng(n);
  std::generate(input_host.begin(), input_host.end(), rng);

  std::vector<int> inits_host(num_tiles);
  std::generate(inits_host.begin(), inits_host.end(), rng);

  // Copy the data to the GPU.
  mem_t<int> input_device = to_mem(input_host, context);
  mem_t<int> inits_device = to_mem(inits_host, context);

  mem_t<int> output_device(n, context);

  auto bandwidth = measure_bandwidth_of_invocation_in_gigabytes_per_second(100, 2 * sizeof(int) * n, [&]
  {
    inclusive_scan_tiles<group_size,grain_size>(input_device.data(), input_device.size(), inits_device.data(), output_device.data(), plus_t<int>(), context);
  });

  std::cout << "Mean bandwidth: " << bandwidth << " GB/s " << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

