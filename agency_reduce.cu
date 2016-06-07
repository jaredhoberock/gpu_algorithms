#include <iostream>
#include <moderngpu/memory.hxx>      // for mem_t.
#include <agency/agency.hpp>
#include <agency/experimental/stride.hpp>
#include <agency/experimental/span.hpp>
#include <agency/experimental/array.hpp>
#include <agency/experimental/optional.hpp>
#include <agency/experimental/chunk.hpp>
#include <agency/cuda.hpp>
#include "measure_bandwidth_of_invocation.hpp"
#include "bound.hpp"
#include "algorithm.hpp"
#include "collective_ptr.hpp"
#include <cstdio>

template<size_t block_size, size_t grain_size = 1>
auto static_grid(int num_blocks) ->
  decltype(agency::cuda::par(num_blocks, agency::cuda::experimental::static_concurrent_execution_policy<block_size, grain_size>()))
{
  return agency::cuda::par(num_blocks, agency::cuda::experimental::static_concurrent_execution_policy<block_size, grain_size>());
}

template<size_t block_size, size_t grain_size = 1>
using static_grid_agent = agency::parallel_group<agency::experimental::static_concurrent_agent<block_size, grain_size>>;

template<typename launch_arg_t = mgpu::empty_t, typename input_it,  typename output_it, class BinaryOperation>
void my_reduce(input_it input, int count, output_it reduction, BinaryOperation binary_op, mgpu::context_t& context)
{
  using namespace mgpu;
  using namespace agency::experimental;

  constexpr int group_size = 128;
  constexpr int grain_size = 8;
  constexpr int chunk_size = group_size * grain_size;
  size_t num_groups = (count + chunk_size - 1) / chunk_size;

  typedef typename std::iterator_traits<input_it>::value_type T;
  mem_t<T> partials(num_groups, context);

  auto partials_view = span<T>(partials.data(), num_groups);
  auto input_view = span<T>(input, count);

  auto k = [=] __device__ (static_grid_agent<128,8>& self)
  {
    int group_rank = self.outer().rank();

    // find this group's chunk of the input
    auto our_chunk = chunk(input_view, chunk_size)[group_rank];
    
    // cooperatively reduce across the group
    // XXX we pay a small penalty for returning the result to the entire group
    //     instead of exclusively rank 0
    auto result = uninitialized_reduce(self.inner(), our_chunk, binary_op);

    if(self.inner().elect())
    {
      if(num_groups > 1)
      {
        partials_view[group_rank] = result;
      }
      else
      {
        *reduction = result;
      }
    }
  };

  agency::bulk_invoke(static_grid<128,8>(num_groups), k);

  // Recursively call reduce until there's just one scalar.
  if(num_groups > 1)
  {
    //my_reduce<launch_params_t<512, 4> >(partials_view.begin(), num_groups, reduction, binary_op, context);
    my_reduce<launch_params_t<128, 8> >(partials_view.begin(), num_groups, reduction, binary_op, context);
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

