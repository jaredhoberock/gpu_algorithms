#include <iostream>
#include <moderngpu/transform.hxx>   // for cta_launch.
#include <moderngpu/memory.hxx>      // for mem_t.
#include <moderngpu/cta_reduce.hxx>
#include <agency/agency.hpp>
#include <agency/experimental/strided_view.hpp>
#include <agency/experimental/span.hpp>
#include <agency/experimental/array.hpp>
#include <agency/experimental/optional.hpp>
#include <agency/cuda.hpp>
#include "measure_bandwidth_of_invocation.hpp"
#include "bound.hpp"
#include "algorithm.hpp"
#include <cstdio>

auto grid(int num_blocks, int num_threads) ->
  decltype(agency::cuda::par(num_blocks, agency::cuda::con(num_threads)))
{
  return agency::cuda::par(num_blocks, agency::cuda::con(num_threads));
}

using grid_agent = agency::parallel_group<agency::cuda::concurrent_agent>;


template<int nt, typename type_t>
struct my_cta_reduce_t
{
  enum
  { 
    num_participating_agents = mgpu::min(nt, (int)mgpu::warp_size), 
    num_passes = mgpu::s_log2(num_participating_agents),
    num_sequential_sums_per_agent = nt / num_participating_agents 
  };

  static_assert(0 == nt % mgpu::warp_size, "cta_reduce_t requires num threads to be a multiple of warp_size (32)");

  using storage_t = agency::experimental::array<type_t, mgpu::max(nt, 2 * num_participating_agents)>;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300

  typedef mgpu::shfl_reduce_t<type_t, num_participating_agents> group_reduce_t;

  template<typename op_t = mgpu::plus_t<type_t> >
  __device__
  agency::experimental::optional<type_t> reduce_and_elect(int tid, agency::experimental::optional<type_t> partial_sum, storage_t& storage, int count = nt, op_t op = op_t()) const
  {
    // store partial sum to storage
    if(tid < count)
    {
      storage[tid] = *partial_sum;
    }

    using namespace agency::experimental;
    auto partial_sums = span<type_t>(storage.data(), count);
    __syncthreads();

    if(tid < num_participating_agents)
    {
      // stride through the input and compute a partial sum per agent
      auto my_partial_sums = strided(drop(partial_sums, tid), (int)num_participating_agents);

      partial_sum = ::uninitialized_reduce(bound<num_sequential_sums_per_agent>(), my_partial_sums, op);

      // Cooperative reduction.
      partial_sum = group_reduce_t().reduce(tid, *partial_sum, min(count, (int)num_participating_agents), op);
    }
    __syncthreads();

    return tid == 0 ? partial_sum : agency::experimental::nullopt;
  }

#else

  template<typename op_t = mgpu::plus_t<type_t> >
  __device__
  agency::experimental::optional<type_t> reduce_and_elect(int tid, agency::experimental::optional<type_t> partial_sum, storage_t& storage, int count = nt, op_t op = op_t(), bool broadcast = true) const
  {
    // store partial sum to storage
    if(tid < count)
    {
      storage[tid] = *partial_sum;
    }

    using namespace agency::experimental;
    auto partial_sums = span<type_t>(storage.data(), count);
    __syncthreads();

    if(tid < num_participating_agents)
    {
      // stride through the input and compute a partial sum per agent
      auto my_partial_sums = strided(drop(partial_sums, tid), (int)num_participating_agents);

      partial_sum = ::uninitialized_reduce(bound<num_sequential_sums_per_agent>(), my_partial_sums, op);

      if(partial_sum)
      {
        storage[tid] = *partial_sum;
      }
    }
    __syncthreads();

    int count2 = min(count, int(num_participating_agents));
    int first = (1 & num_passes) ? num_participating_agents : 0;
    if(tid < num_participating_agents && partial_sum)
    {
      storage[first + tid] = *partial_sum;
    }
    __syncthreads();


    int offset = 1;
    for(int pass = 0; pass < num_passes; ++pass, offset *= 2)
    {
      if(tid < num_participating_agents)
      {
        if(tid + offset < count2) 
        {
          partial_sum = op(*partial_sum, storage[first + offset + tid]);
        }

        first = num_participating_agents - first;
        storage[first + tid] = *partial_sum;
      }
      __syncthreads();
    }

    return tid == 0 ? partial_sum : agency::experimental::nullopt;
  }

#endif

  template<typename op_t = mgpu::plus_t<type_t> >
  __device__
  type_t reduce(int tid, agency::experimental::optional<type_t> partial_sum, storage_t& storage, int count = nt, op_t op = op_t()) const
  {
    auto result = reduce_and_elect(tid, partial_sum, storage, count, op);

    // XXX we're using inside knowledge that reduce_and_elect() always elects tid == 0
    if(tid == 0)
    {
      storage[0] = *result;
    }

    __syncthreads();

    return storage[0];
  }

};


template<typename launch_arg_t = mgpu::empty_t, typename input_it,  typename output_it, typename op_t>
void my_reduce(input_it input, int count, output_it reduction, op_t op, mgpu::context_t& context)
{
  using namespace mgpu;
  using namespace agency::experimental;

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_params_t<128, 8>
  >::type_t launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);
  int num_threads = launch_t::cta_dim(context).nt;
  mem_t<type_t> partials(num_ctas, context);
  type_t* partials_data = partials.data();

  auto k = [=] __device__ (int tid, int cta)
  {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
    typedef my_cta_reduce_t<nt, type_t> reduce_t;
    __shared__ typename reduce_t::storage_t shared_reduce;

    // Load the data for the first tile for each cta.
    range_t tile = get_tile(cta, nv, count);

    // stride through the input and compute a partial sum per thread
    span<type_t> our_span(input + tile.begin, input + tile.end);
    auto my_values = strided(drop(our_span, tid), size_t(nt));

    // we don't have an initializer for the agent's sum, so use uninitialized_reduce
    auto partial_sum = uninitialized_reduce(bound<vt>(), my_values, op);

    // Reduce to a scalar per CTA.
    int num_partials = min(tile.count(), (int)nt);
    auto result = reduce_t().reduce_and_elect(tid, partial_sum, shared_reduce, num_partials, op);

    if(result)
    {
      if(1 == num_ctas) *reduction = *result;
      else partials_data[cta] = *result;
    }
  };

  agency::bulk_invoke(grid(num_ctas, num_threads), [=] __device__ (grid_agent& self)
  {
    k(self.inner().index(), self.outer().index());
  });

  // Recursively call reduce until there's just one scalar.
  if(num_ctas > 1)
  {
    my_reduce<launch_params_t<512, 4> >(partials_data, num_ctas, reduction, op, context);
  }
}

int main(int argc, char** argv)
{
  using namespace mgpu;

  standard_context_t context;

  size_t n = (1 << 30) + 13;

  // Prepare the fibonacci numbers on the host.
  std::vector<int> input_host(n);
  for(int i = 0; i < input_host.size(); ++i)
    input_host[i] = (i + 1) * (i + 1);

  // Copy the data to the GPU.
  mem_t<int> input_device = to_mem(input_host, context);

  // Call our simple reduce.
  mem_t<int> output_device(1, context);
  my_reduce(input_device.data(), input_device.size(), output_device.data(), plus_t<int>(), context);

  // Get the reduction.
  std::vector<int> output_host = from_mem(output_device);

  // compare to reference
  assert(std::accumulate(input_host.begin(), input_host.end(), 0, std::plus<int>()) == output_host[0]);

  auto bandwidth = measure_bandwidth_of_invocation_in_gigabytes_per_second(100, sizeof(int) * n, [&]
  {
    my_reduce(input_device.data(), input_device.size(), output_device.data(), plus_t<int>(), context);
  });

  std::cout << "Mean bandwidth: " << bandwidth << " GB/s " << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

