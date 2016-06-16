#include <numeric>
#include <iostream>
#include <random>
#include <algorithm>

#include <moderngpu/transform.hxx>   // for cta_launch.
#include <moderngpu/memory.hxx>      // for mem_t.
#include <moderngpu/cta_scan.hxx>
#include <agency/agency.hpp>
#include <agency/experimental/span.hpp>
#include <agency/experimental/chunk.hpp>
#include <agency/experimental/short_vector.hpp>
#include <agency/cuda.hpp>
#include "measure_bandwidth_of_invocation.hpp"
#include "algorithm.hpp"
#include "algorithm/copy.hpp"
#include "algorithm/inclusive_scan.hpp"
#include "algorithm/exclusive_scan.hpp"
#include "scanning_barrier.hpp"


auto grid(int num_blocks, int num_threads) ->
  decltype(agency::cuda::par(num_blocks, agency::cuda::con(num_threads)))
{
  return agency::cuda::par(num_blocks, agency::cuda::con(num_threads));
}

using grid_agent = agency::parallel_group<agency::concurrent_agent>;


template<size_t block_size, size_t grain_size = 1>
auto static_grid(int num_blocks) ->
  decltype(agency::cuda::par(num_blocks, agency::cuda::experimental::static_concurrent_execution_policy<block_size, grain_size>()))
{
  return agency::cuda::par(num_blocks, agency::cuda::experimental::static_concurrent_execution_policy<block_size, grain_size>());
}

template<size_t block_size, size_t grain_size = 1>
using static_grid_agent = agency::parallel_group<agency::experimental::static_concurrent_agent<block_size, grain_size>>;

using namespace mgpu;


template<bool exclusive, size_t group_size, size_t grain_size, class Range1, class Range2, class BinaryOperation, class T>
__device__
T scan(agency::experimental::static_concurrent_agent<group_size,grain_size>& self, Range1&& in, Range2&& out, BinaryOperation binary_op, T init)
{
  using namespace agency::experimental;

  constexpr size_t tile_size = group_size * grain_size;

  int tid = self.rank();
  
  typedef my_cta_scan_t<group_size, T> scan_t;
  
  __shared__ union {
    typename scan_t::storage_t scan;
    T values[tile_size];
  } shared;
  
  // XXX cooperatively call scan
  //     the implementation is below

  auto view_of_input_tiles  = chunk(in, tile_size);
  auto view_of_output_tiles = chunk(out, tile_size);
  
  for(int tile = 0; tile < view_of_input_tiles.size(); ++tile)
  {
    auto view_of_current_input_tile = view_of_input_tiles[tile];
  
    // cooperatively copy the tile from input into shared memory 
    auto view_of_shared = span<T>(shared.values, view_of_current_input_tile.size());
    bounded_copy(self, view_of_current_input_tile, view_of_shared);
  
    // partition shared memory into subtiles of grain_size width
    auto view_of_shared_subtiles = chunk(view_of_shared, grain_size);
  
    // each thread gets a view of its subtile, unless we've reached the end of the input, in which case it gets an empty view
    auto view_of_shared_subtile = tid < view_of_shared_subtiles.size() ? view_of_shared_subtiles[tid] : view_of_shared_subtiles.empty_chunk();
  
    // each thread copies its subtile into an in-register local array
    short_vector<T,grain_size> local_subtile = view_of_shared_subtile;
  
    // each thread sums its subtile
    T summand;
    if(local_subtile.size())
    {
      summand = accumulate_nonempty(bound<grain_size>(), local_subtile, binary_op);
    }
  
    self.wait();
  
    // compute the prefix sum of the per-thread sums 
    // this prefix sum is the "spine"
    my_scan_result_t<T> scan_of_spine = scan_t().exclusive_scan(tid, summand, shared.scan, view_of_shared_subtiles.size(), init, binary_op);
  
    // each thread computes the prefix sum of its subtile and puts the result directly into shared memory
    if(exclusive)
    {
      exclusive_scan(bound<grain_size>(), local_subtile, view_of_shared_subtile, scan_of_spine.scan, binary_op);
    }
    else
    {
      inclusive_scan(bound<grain_size>(), local_subtile, view_of_shared_subtile, binary_op, scan_of_spine.scan);
    }
  
    self.wait();
  
    // cooperatively copy the tile from shared memory to the output
    auto view_of_current_output_tile = view_of_output_tiles[tile];
    bounded_copy(self, view_of_shared, view_of_current_output_tile);
  
    // update the current carry
    init = scan_of_spine.reduction;
  }

  return init;
}


template<size_t group_size, size_t grain_size, class Range1, class Range2, class BinaryOperation, class T>
__device__
T inclusive_scan(agency::experimental::static_concurrent_agent<group_size,grain_size>& self, Range1&& in, Range2&& out, BinaryOperation binary_op, T init)
{
  return scan<false>(self, in, out, binary_op, init);
}


template<class Range1, class Range2, class BinaryOperation, class T>
void inclusive_scan(Range1&& in, Range2&& out, BinaryOperation binary_op, T init)
{
  using namespace agency::experimental;

  constexpr size_t group_size = 512;
  constexpr size_t grain_size = 3;
  constexpr size_t tile_size = group_size;

  auto input_view  = all(in);
  auto output_view = all(out);

  size_t num_ctas = (input_view.size() + tile_size - 1) / tile_size;

  //if(num_ctas > 8)
  //{
  //  auto upsweep_kernel = [=] __device__ (static_grid_agent<group_size,grain_size>& self)
  //  {
  //    int group_rank = self.outer().rank();

  //    // find this group's chunk of the input
  //    constexpr size_t tile_size = group_size * grain_size;
  //    auto view_of_this_groups_chunk = chunk(input_view, tile_size)[group_rank];
  //    
  //    // cooperatively reduce across the group and elect an agent to receive the result
  //    // XXX this should respect noncommutativity and reduce() doesn't do that
  //    auto result = uninitialized_reduce_and_elect(self.inner(), view_of_this_groups_chunk, op);

  //    if(result)
  //    {
  //      partials_view[group_rank] = *result;
  //    }
  //  };

  //  agency::bulk_invoke(static_grid<group_size,grain_size>(1), upsweep_kernel);
  //}
  //else
  //{
    auto kernel = [=] __device__ (static_grid_agent<group_size,grain_size>& self)
    {
      inclusive_scan(self.inner(), input_view, output_view, binary_op, init);
    };

    agency::bulk_invoke(static_grid<group_size,grain_size>(1), kernel);
  //}
}


bool validate(size_t n, standard_context_t& context)
{
  using namespace agency::experimental;

  std::default_random_engine rng(n);
  std::vector<int> input_host(n);
  std::generate(input_host.begin(), input_host.end(), rng);
  
  int init = rng();
  
  // Copy the data to the GPU.
  mem_t<int> input_device = to_mem(input_host, context);
  
  // Call our exclusive scan.
  mem_t<int> output_device(n, context);
  inclusive_scan(span<int>(input_device.data(), input_device.size()), span<int>(output_device.data(), output_device.size()), plus_t<int>(), init);
  
  // Get the result.
  std::vector<int> output_host = from_mem(output_device);
  
  // compare to reference
  std::vector<int> reference = input_host;
  if(reference.size() > 0)
  {
    reference.front() += init;
  }
  
  std::partial_sum(reference.begin(), reference.end(), reference.begin(), plus_t<int>());
  
  if(reference != output_host)
  {
    std::cout << "mismatch at n: " << n << std::endl;
  
    for(auto x : output_host)
    {
      std::cout << x << " ";
    }
    std::cout << std::endl;
  
    return false;
  }

  return true;
}


int main()
{
  using namespace agency::experimental;

  standard_context_t context;

  for(size_t n = 0; n < 100 * 128 * 2; ++n)
  {
    assert(validate(n, context));
  }

  size_t n = 8 * 128 * 11;

  std::vector<int> input_host(n, 1);
  mem_t<int> output_device(n, context);

  mem_t<int> input_device = to_mem(input_host, context);

  auto bandwidth = measure_bandwidth_of_invocation_in_gigabytes_per_second(100, 2 * sizeof(int) * n, [&]
  {
    inclusive_scan(span<int>(input_device.data(), input_device.size()), span<int>(output_device.data(), output_device.size()), plus_t<int>(), 0);
  });

  std::cout << "Mean bandwidth: " << bandwidth << " GB/s " << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

