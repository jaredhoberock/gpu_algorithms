#include <numeric>
#include <iostream>

#include <moderngpu/transform.hxx>   // for cta_launch.
#include <moderngpu/memory.hxx>      // for mem_t.
#include <moderngpu/cta_scan.hxx>
#include <agency/agency.hpp>
#include <agency/experimental/span.hpp>
#include <agency/experimental/chunk.hpp>
#include <agency/cuda.hpp>
#include "measure_bandwidth_of_invocation.hpp"
#include "algorithm.hpp"
#include "algorithm/copy.hpp"
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


template<int nt, int vt, int vt0 = vt, typename it_t>
__device__
array_t<typename std::iterator_traits<it_t>::value_type, vt> 
my_mem_to_reg_strided(it_t mem, int tid, int count)
{
  typedef typename std::iterator_traits<it_t>::value_type type_t;

  array_t<type_t, vt> x;
  strided_iterate<nt, vt, vt0>([&](int i, int j)
  { 
    x[i] = mem[j]; 
  },
  tid,
  count
  );

  return x;
}


template<int nt, int vt, int vt0 = vt, typename type_t, typename it_t, int shared_size>
__device__ agency::experimental::array<type_t, vt>
my_mem_to_reg_thread(it_t mem, int tid, int count, type_t (&shared)[shared_size])
{
  array_t<type_t, vt> x = my_mem_to_reg_strided<nt, vt, vt0>(mem, tid, count);

  // XXX this call synchronizes
  reg_to_shared_strided<nt, vt>(x, tid, shared);

  agency::experimental::array<type_t, vt> y;

  for(size_t i = 0; i < vt; ++i)
  {
    y[i] = shared[tid * vt + i];
  }

  __syncthreads();

  return y;
}

template<mgpu::scan_type_t scan_type = mgpu::scan_type_exc, 
  typename launch_arg_t = empty_t, typename input_it, 
  typename output_it, typename op_t, typename reduction_it>
void my_scan_event(input_it input, int count, output_it output, op_t op, reduction_it reduction, context_t& context)
{
  using namespace agency::experimental;

  constexpr size_t group_size = 128;
  constexpr size_t grain_size = 11;

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_52_cta<group_size, grain_size>
    >
  >::type_t launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;
  using T = type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);
  int num_threads = launch_t::cta_dim(context).nt;

  auto input_view  = span<T>(input, count);
  auto output_view = span<T>(output, count);

  if(num_ctas > 8)
  {
    mem_t<type_t> partials(num_ctas, context);
    type_t* partials_data = partials.data();

    ////////////////////////////////////////////////////////////////////////////
    // Upsweep phase. Reduce each tile to a scalar and store to partials.
    auto partials_view = span<T>(partials.data(), num_ctas);

    auto upsweep_k = [=] __device__ (static_grid_agent<group_size,grain_size>& self)
    {
      int group_rank = self.outer().rank();

      // find this group's chunk of the input
      constexpr size_t chunk_size = group_size * grain_size;
      auto view_of_this_groups_chunk = chunk(input_view, chunk_size)[group_rank];
      
      // cooperatively reduce across the group and elect an agent to receive the result
      // XXX this should respect noncommutativity and reduce() doesn't do that
      auto result = uninitialized_reduce_and_elect(self.inner(), view_of_this_groups_chunk, op);

      if(result)
      {
        partials_view[group_rank] = *result;
      }
    };

    agency::bulk_invoke(static_grid<group_size,grain_size>(num_ctas), upsweep_k);

    ////////////////////////////////////////////////////////////////////////////
    // Spine phase. Recursively call scan on the CTA partials.

    ::my_scan_event<mgpu::scan_type_exc>(partials_data, num_ctas, partials_data,
      op, reduction, context);

    ////////////////////////////////////////////////////////////////////////////
    // Downsweep phase. Perform an intra-tile scan and add the scan of the 
    // partials as carry-in.

    auto downsweep_k = [=] MGPU_DEVICE(int tid, int cta) {
      typedef typename launch_t::sm_ptx params_t;
      enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
      typedef my_cta_scan_t<nt, type_t> scan_t;

      __shared__ union {
        typename scan_t::storage_t scan;
        type_t values[nv];
      } shared;

      // Load a tile to register in thread order.
      range_t tile = get_tile(cta, nv, count);
      array<type_t, vt> local_subtile = my_mem_to_reg_thread<nt, vt>(input + tile.begin, tid, tile.count(), shared.values);

      // Scan the array with carry-in from the partials.
      if(scan_type == mgpu::scan_type_exc)
      {
        local_subtile = scan_t().exclusive_scan(tid, local_subtile, shared.scan, partials_data[cta], tile.count(), op).scan;
      }
      else
      {
        local_subtile = scan_t().inclusive_scan(tid, local_subtile, shared.scan, partials_data[cta], tile.count(), op).scan;
      }

      // XXX eliminate this temporary
      array_t<type_t, vt> temp;
      sequential_bounded_copy<vt>(local_subtile, temp);

      // Store the scanned values to the output.
      reg_to_mem_thread<nt, vt>(temp, tid, tile.count(), output + tile.begin, shared.values);
    };

    agency::bulk_invoke(grid(num_ctas, num_threads), [=] __device__ (grid_agent& self)
    {
      downsweep_k(self.inner().index(), self.outer().index());
    });
  
  } else {

    ////////////////////////////////////////////////////////////////////////////
    // Small input specialization. This is the non-recursive branch.

    typedef launch_params_t<512, 3> spine_params_t;
    auto spine_k = [=] MGPU_DEVICE(static_grid_agent<512,3>& self)
    {
      int tid = self.inner().rank();
     
      enum { nt = spine_params_t::nt, vt = spine_params_t::vt, nv = nt * vt };
      typedef my_cta_scan_t<nt, type_t> scan_t;

      __shared__ union {
        typename scan_t::storage_t scan;
        type_t values[nv];
      } shared;

      // XXX cooperatively call scan
      //     the implementation is below

      auto view_of_input_tiles  = chunk(input_view, nv);
      auto view_of_output_tiles = chunk(output_view, nv);

      //auto view_of_shared = span<type_t>(shared.values, nv);

      type_t carry_in = 0;
      //for(int cur = 0; cur < count; cur += nv)
      for(int tile = 0; tile < view_of_input_tiles.size(); ++tile)
      {
        // Cooperatively load values into register.
        //int count2 = min<int>(count - cur, nv);

        //auto view_of_current_input_tile = view_of_input_tiles[cur];
        auto view_of_current_input_tile = view_of_input_tiles[tile];

        // XXX 1. cooperatively copy the tile into shared memory 
        // copy the current tile into smem
        auto view_of_shared = span<type_t>(shared.values, view_of_current_input_tile.size());
        bounded_copy(self.inner(), view_of_current_input_tile, view_of_shared);

        // partition shared memory into subtiles
        auto view_of_subtile = chunk(view_of_shared, vt)[tid];

        // XXX 2. each thread copies its subtile into a local array
        // copy subtile into local array
        array<type_t, vt> local_subtile;
        sequential_bounded_copy<vt>(chunk(view_of_shared, vt)[tid], local_subtile);

        my_scan_result_t<type_t, vt> result;
        if(scan_type == mgpu::scan_type_exc)
        {
          //result = scan_t().exclusive_scan(tid, temp, shared.scan, carry_in, count2, op);
          result = scan_t().exclusive_scan(tid, local_subtile, shared.scan, carry_in, view_of_shared.size(), op);
        }
        else
        {
          //result = scan_t().inclusive_scan(tid, temp, shared.scan, carry_in, count2, op);
          result = scan_t().inclusive_scan(tid, local_subtile, shared.scan, carry_in, view_of_shared.size(), op);
        }

        // XXX 4. each thread copies its subtile back into shared memory
        sequential_bounded_copy<vt>(result.scan, local_subtile);

        sequential_bounded_copy<vt>(local_subtile, chunk(view_of_shared, vt)[tid]);

        // XXX 5. cooperatively copy the tile from shared memory to the result
        //auto view_of_current_output_tile = view_of_output_tiles[cur];
        auto view_of_current_output_tile = view_of_output_tiles[tile];
        bounded_copy(self.inner(), view_of_shared, view_of_current_output_tile);

        //// Store the scanned values back to global memory.
        //reg_to_mem_thread<nt, vt>(result.scan, tid, count2, output + cur, shared.values);
        
        // Roll the reduction into carry_in.
        carry_in = result.reduction;
      }

      // Store the carry-out to the reduction pointer. This may be a
      // discard_iterator_t if no reduction is wanted.
      if(!tid)
        *reduction = carry_in;
    };

    agency::bulk_invoke(static_grid<512,3>(1), spine_k);
  }
}

template<typename launch_arg_t = empty_t, typename input_it, typename output_it, typename op_t>
void exclusive_scan(input_it input, int count, output_it output, op_t op, context_t& context)
{
  return my_scan_event<mgpu::scan_type_exc, launch_arg_t>(input, count, output, op, discard_iterator_t<typename std::iterator_traits<output_it>::value_type>(), context);
}

template<typename launch_arg_t = empty_t, typename input_it, typename output_it, typename op_t>
void inclusive_scan(input_it input, int count, output_it output, op_t op, context_t& context)
{
  return my_scan_event<mgpu::scan_type_inc, launch_arg_t>(input, count, output, op, discard_iterator_t<typename std::iterator_traits<output_it>::value_type>(), context);
}


int main()
{
  standard_context_t context;

  size_t n = 1 << 30;
  //size_t n = 128 * 11 * 7;
  //size_t n = 128 * 11 * 1;

  std::vector<int> input_host(n, 1);

  // Copy the data to the GPU.
  mem_t<int> input_device = to_mem(input_host, context);

  // Call our exclusive scan.
  mem_t<int> output_device(n, context);
  //exclusive_scan(input_device.data(), input_device.size(), output_device.data(), plus_t<int>(), context);
  inclusive_scan(input_device.data(), input_device.size(), output_device.data(), plus_t<int>(), context);

  // Get the result.
  std::vector<int> output_host = from_mem(output_device);

  // compare to reference
  std::vector<int> reference(n);
  std::iota(reference.begin(), reference.end(), 1);
  assert(reference == output_host);

  auto bandwidth = measure_bandwidth_of_invocation_in_gigabytes_per_second(100, 2 * sizeof(int) * n, [&]
  {
    //exclusive_scan(input_device.data(), input_device.size(), output_device.data(), plus_t<int>(), context);
    inclusive_scan(input_device.data(), input_device.size(), output_device.data(), plus_t<int>(), context);
  });

  std::cout << "Mean bandwidth: " << bandwidth << " GB/s " << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

