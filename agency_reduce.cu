#include <iostream>
#include <moderngpu/memory.hxx>      // for mem_t.
#include <agency/agency.hpp>
#include <agency/experimental/strided_view.hpp>
#include <agency/experimental/span.hpp>
#include <agency/experimental/array.hpp>
#include <agency/experimental/optional.hpp>
#include <agency/experimental/chunk.hpp>
#include <agency/cuda.hpp>
#include "measure_bandwidth_of_invocation.hpp"
#include "bound.hpp"
#include "algorithm.hpp"
#include "reducing_barrier.hpp"
#include <cstdio>

auto grid(int num_blocks, int num_threads) ->
  decltype(agency::cuda::par(num_blocks, agency::cuda::con(num_threads)))
{
  return agency::cuda::par(num_blocks, agency::cuda::con(num_threads));
}

using grid_agent = agency::parallel_group<agency::cuda::concurrent_agent>;

template<typename launch_arg_t = mgpu::empty_t, typename input_it,  typename output_it, class BinaryOperation>
void my_reduce(input_it input, int count, output_it reduction, BinaryOperation binary_op, mgpu::context_t& context)
{
  using namespace mgpu;
  using namespace agency::experimental;

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_params_t<128, 8>
  >::type_t launch_t;

  typedef typename std::iterator_traits<input_it>::value_type T;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);
  int num_threads = launch_t::cta_dim(context).nt;
  mem_t<T> partials(num_ctas, context);
  T* partials_data = partials.data();

  auto input_view = span<T>(input, count);

  auto k = [=] __device__ (int agent_idx, int block_idx)
  {
    typedef typename launch_t::sm_ptx params_t;

    constexpr int num_threads = params_t::nt;
    constexpr int grainsize = params_t::vt;
    constexpr int tile_size = num_threads * grainsize;

    // find this group's chunk of the input
    auto our_chunk = chunk(input_view, tile_size)[block_idx];
    
    // each agent strides through its group's chunk of the input...
    auto my_values = strided(drop(our_chunk, agent_idx), size_t(num_threads));

    // ...and sequentially computes a partial sum
    auto partial_sum = uninitialized_reduce(bound<grainsize>(), my_values, binary_op);

    // the entire group cooperatively reduces the partial sums
    int num_partials = min<int>(our_chunk.size(), (int)num_threads);

    __shared__ reducing_barrier<T, num_threads> barrier;
    auto result = barrier.reduce_and_wait_and_elect(agent_idx, partial_sum, num_partials, binary_op);

    if(result)
    {
      if(num_ctas > 1)
      {
        partials_data[block_idx] = *result;
      }
      else
      {
        *reduction = *result;
      }
    }
  };

  agency::bulk_invoke(grid(num_ctas, num_threads), [=] __device__ (grid_agent& self)
  {
    k(self.inner().index(), self.outer().index());
  });

  // Recursively call reduce until there's just one scalar.
  if(num_ctas > 1)
  {
    my_reduce<launch_params_t<512, 4> >(partials_data, num_ctas, reduction, binary_op, context);
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

