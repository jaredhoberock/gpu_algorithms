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
#include "myloadstore.hpp"
#include "my_cta_scan.hpp"


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


// XXX abstracting this into a function introduces a register
template<bool exclusive, size_t group_size, size_t grain_size, class Range1, class Range2, class T, class BinaryOperation>
__device__
T bounded_scan_tile(agency::experimental::static_concurrent_agent<group_size,grain_size>& self, const Range1& input_tile, Range2&& output_tile, T init, BinaryOperation binary_op)
{
  using namespace agency::experimental;

  static_assert(!exclusive, "exclusive scan unimplemented!");
  constexpr size_t tile_size = group_size * grain_size;

  int tid = self.index();

  typedef my_cta_scan_t<group_size, T> scan_t;

  __shared__ union {
    typename scan_t::storage_t scan;
    array<T,tile_size> values;
  } shared;

  // XXX should simply call take() here
  auto shared_tile = span<T>(shared.values.data(), input_tile.size());

  // copy the tile from global memory into shared memory
  bounded_copy<group_size,grain_size>(tid, input_tile, shared_tile);

  // XXX should call something like to_container() here
  //     statically-sized ranges would be copied into an array<T,N>
  //     bounded-sized ranges would be copied into a short_vector<T,N>
  auto view_of_subtile = chunk(shared_tile, grain_size).chunk_or_empty(tid);

  // copy the subtile from shared memory into registers
  short_vector<T,grain_size> x = view_of_subtile;

  __syncthreads();

  // each agent does an in-place inclusive of its local array
  ::inclusive_scan<grain_size>(x, x, binary_op);

  // scan the thread-local reductions for a carry-in for each thread
  auto cooperative_result = scan_t().exclusive_scan(tid, x[grain_size-1], shared.scan, div_up<int>(input_tile.size(), grain_size), init, binary_op);

  // to produce the final inclusive scan, add the thread's carry-in
  bounded_for_loop<grain_size>(x.size(), [&](int i)
  {
    x[i] = binary_op(cooperative_result.scan, x[i]);
  });

  // copy the subtile from registers into shared memory
  sequential_bounded_copy<grain_size>(x, view_of_subtile);

  __syncthreads();

  // copy the tile shared memory into global memory
  bounded_copy<group_size,grain_size>(tid, shared_tile, output_tile);

  return cooperative_result.reduction;
}


// XXX abstracting this into a function introduces a register
template<bool exclusive, size_t group_size, size_t grain_size, class Range1, class Range2, class T, class BinaryOperation>
__device__
T static_scan_tile(agency::experimental::static_concurrent_agent<group_size,grain_size>& self, const Range1& input_tile, Range2&& output_tile, T init, BinaryOperation binary_op)
{
  using namespace agency::experimental;

  static_assert(!exclusive, "exclusive scan unimplemented!");
  constexpr size_t tile_size = group_size * grain_size;

  int tid = self.index();

  typedef my_cta_scan_t<group_size, T> scan_t;

  __shared__ union {
    typename scan_t::storage_t scan;
    array<T,tile_size> values;
  } shared;

  // XXX remember that the last, partial tile will not necessarily be tile_size!!!
  //auto shared_tile = span<T>(shared.values, input_tile.size());
  auto shared_tile = span<T>(shared.values.data(), tile_size);

  bounded_copy<group_size,grain_size>(tid, input_tile, shared_tile);

  // XXX remember that the last, partial tile will not necessarily have a chunk for each thread!!!
  auto view_of_subtile = chunk(shared_tile, grain_size)[tid];
  array<T,grain_size> x;

  // XXX this sequential_bounded_copy() call adds 5 registers over the use of just a static_for_loop
  //     because sequential_bounded_copy works for dynamically-sized copies
  // XXX 32 registers
  //sequential_bounded_copy<grain_size>(view_of_subtile, x);
  // XXX 5 fewer registers if we use counted() to fix the size of the copy
  //     5 fewer registers
  sequential_bounded_copy<grain_size>(counted(view_of_subtile, grain_size), x);

  __syncthreads();

  // each agent does an in-place inclusive of its local array
  ::inclusive_scan<grain_size>(x, x, binary_op);

  // scan the thread-local reductions for a carry-in for each thread
  auto cooperative_result = scan_t().exclusive_scan(tid, x[grain_size-1], shared.scan, div_up<int>(input_tile.size(), grain_size), init, binary_op);

  // to produce the final inclusive scan, add the thread's carry-in
  // XXX remember that the last, partial tile will not necessarily have a full subtile!!!
  static_for_loop<grain_size>([&](int i)
  {
    x[i] = binary_op(cooperative_result.scan, x[i]);
  });

  // XXX this adds 1 register over the use of just a static_for_loop
  //     probably because sequential_bounded_copy works for dynamically-sized copies
  //     28 registers
  sequential_bounded_copy<grain_size>(x, view_of_subtile);

  __syncthreads();

  bounded_copy<group_size,grain_size>(tid, shared_tile, output_tile);

  return cooperative_result.reduction;
}


template<size_t group_size, size_t grain_size, typename input_it1, typename input_it2, typename output_it, typename op_t>
void inclusive_scan_tiles(input_it1 input, int count, input_it2 inits, output_it output, op_t binary_op, context_t& context)
{
  constexpr size_t tile_size = group_size * grain_size;

  using launch_t = launch_box_t<
    arch_52_cta<group_size, grain_size>
  >;

  typedef typename std::iterator_traits<input_it1>::value_type T;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);

  auto input_view  = span<T>(input, count);
  auto output_view = span<T>(output, count);

  auto input_tiles = chunk(input_view, tile_size);
  auto output_tiles = chunk(output_view, tile_size);

  auto downsweep_k = [=] __device__ (static_grid_agent<group_size,grain_size>& self)
  {
    int cta = self.outer().index();
    auto input_tile = chunk(input_view, tile_size)[cta]; 
    auto output_tile = chunk(output_view, tile_size)[cta];
    //static_scan_tile<false>(self.inner(), input_tile, output_tile, inits[cta], binary_op); 
    bounded_scan_tile<false>(self.inner(), input_tile, output_tile, inits[cta], binary_op); 
  };
  agency::bulk_invoke(static_grid<group_size,grain_size>(num_ctas), downsweep_k);
}


int main()
{
  standard_context_t context;

  constexpr size_t group_size = 128;
  constexpr size_t grain_size = 11;
  constexpr size_t tile_size = group_size * grain_size;
  constexpr size_t num_tiles = 1 << 20;
  constexpr size_t n = tile_size * num_tiles;

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

  std::cout << "Mean bandwidth: " << bandwidth << " GB/s " << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

