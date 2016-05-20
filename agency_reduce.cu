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


// XXX seems like this is really only valid for POD
// XXX if we destroy x and properly construct the result, how correct does that make it?
template<class T>
__device__
T shuffle_down(const T& x, int offset, int width)
{ 
  constexpr std::size_t num_words = mgpu::div_up(sizeof(T), sizeof(int));

  union
  {
    int words[num_words];
    T value;
  } u;
  u.value = x;

  for(int i = 0; i < num_words; ++i)
  {
    u.words[i] = __shfl_down(u.words[i], offset, width);
  }

  return u.value;
}


template<class T>
__device__
agency::experimental::optional<T> optionally_shuffle_down(const agency::experimental::optional<T>& x, int offset, int width)
{
  constexpr std::size_t num_words = mgpu::div_up(sizeof(T), sizeof(int));

  union
  {
    int words[num_words];
    T value;
  } u;

  if(x)
  {
    u.value = *x;
  }

  for(int i = 0; i < num_words; ++i)
  {
    u.words[i] = __shfl_down(u.words[i], offset, width);
  }

  // communicate whether or not the words we shuffled came from a valid object
  bool is_valid = x ? true : false;
  is_valid = __shfl_down(is_valid, offset, width);

  return is_valid ? agency::experimental::make_optional(u.value) : agency::experimental::nullopt;
}



// requires __CUDA_ARCH__ >= 300.
// num_threads can be any power-of-two <= warp_size.
// warp_reduce_t returns the reduction only in lane 0.
template<class T, int num_threads>
struct warp_reducing_barrier
{
  static_assert(num_threads <= mgpu::warp_size && mgpu::is_pow2(num_threads), "shfl_reduce_t must operate on a pow2 number of threads <= warp_size (32)");

  template<typename BinaryOperation>
  __device__
  agency::experimental::optional<T> reduce_and_wait_and_elect(int lane, agency::experimental::optional<T> x, int count, BinaryOperation binary_op) const
  {
    if(count == num_threads)
    { 
      for(int pass = 0; pass < mgpu::s_log2(num_threads); ++pass)
      {
        int offset = 1 << pass;
        auto y = shuffle_down(*x, offset, num_threads);
        x = binary_op(*x, y);
      }
    }
    else
    {
      for(int pass = 0; pass < mgpu::s_log2(num_threads); ++pass)
      {
        int offset = 1 << pass;
        auto y = optionally_shuffle_down(x, offset, num_threads);
        if((lane + offset < count) && y) *x = binary_op(*x, *y);
      }
    }

    return (lane == 0) ? x : agency::experimental::nullopt;
  }
};


template<class T, int num_agents>
class reducing_barrier
{
  public:
    static_assert(0 == num_agents % mgpu::warp_size, "num_agents must be a multiple of warp_size (32)");
   
    __device__
    reducing_barrier() = default;

    reducing_barrier(const reducing_barrier&) = delete;

    reducing_barrier(reducing_barrier&&) = delete;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
    template<class BinaryOperation>
    __device__
    agency::experimental::optional<T> reduce_and_wait_and_elect(int agent_rank, const agency::experimental::optional<T>& value, int count, BinaryOperation binary_op)
    {
      auto partial_sum = value;

      // store partial sum to storage
      if(agent_rank < count)
      {
        storage_[agent_rank] = *partial_sum;
      }

      using namespace agency::experimental;
      auto partial_sums = span<T>(storage_.data(), count);
      __syncthreads();

      if(agent_rank < num_participating_agents)
      {
        // stride through the input and compute a partial sum per agent
        auto my_partial_sums = strided(drop(partial_sums, agent_rank), (int)num_participating_agents);

        partial_sum = ::uninitialized_reduce(bound<num_sequential_sums_per_agent>(), my_partial_sums, binary_op);

        // reduce across the warp
        partial_sum = warp_barrier_.reduce_and_wait_and_elect(agent_rank, partial_sum, min(count, (int)num_participating_agents), binary_op);
      }
      __syncthreads();

      return agent_rank == 0 ? partial_sum : agency::experimental::nullopt;
    }

#else

    template<class BinaryOperation>
    __device__
    agency::experimental::optional<T> reduce_and_wait_and_elect(int agent_rank, const agency::experimental::optional<T>& value, int count, BinaryOperation binary_op)
    {
      auto partial_sum = value;

      // store partial sum to storage
      if(agent_rank < count)
      {
        storage_[agent_rank] = *partial_sum;
      }

      using namespace agency::experimental;
      auto partial_sums = span<T>(storage_.data(), count);
      __syncthreads();

      if(agent_rank < num_participating_agents)
      {
        // stride through the input and compute a partial sum per agent
        auto my_partial_sums = strided(drop(partial_sums, agent_rank), (int)num_participating_agents);

        partial_sum = ::uninitialized_reduce(bound<num_sequential_sums_per_agent>(), my_partial_sums, binary_op);

        if(partial_sum)
        {
          storage_[agent_rank] = *partial_sum;
        }
      }
      __syncthreads();

      int count2 = min(count, int(num_participating_agents));
      int first = (1 & num_passes) ? num_participating_agents : 0;
      if(agent_rank < num_participating_agents && partial_sum)
      {
        storage_[first + agent_rank] = *partial_sum;
      }
      __syncthreads();


      int offset = 1;
      for(int pass = 0; pass < num_passes; ++pass, offset *= 2)
      {
        if(agent_rank < num_participating_agents)
        {
          if(agent_rank + offset < count2) 
          {
            partial_sum = binary_op(*partial_sum, storage_[first + offset + agent_rank]);
          }

          first = num_participating_agents - first;
          storage_[first + agent_rank] = *partial_sum;
        }
        __syncthreads();
      }

      return agent_rank == 0 ? partial_sum : agency::experimental::nullopt;
    }

#endif

    template<class BinaryOperation>
    __device__
    T reduce_and_wait(int agent_rank, const agency::experimental::optional<T>& value, int count, BinaryOperation binary_op) const
    {
      auto result = reduce_and_wait_and_elect(agent_rank, value, count, binary_op);

      // XXX we're using inside knowledge that reduce_and_elect() always elects agent_rank == 0
      if(agent_rank == 0)
      {
        storage_[0] = *result;
      }

      __syncthreads();

      return storage_[0];
    }

  private:
    // XXX these should just be constexpr members, but nvcc crashes when i do that
    enum
    { 
      num_participating_agents = mgpu::min(num_agents, (int)mgpu::warp_size), 
      num_passes = mgpu::s_log2(num_participating_agents),
      num_sequential_sums_per_agent = num_agents / num_participating_agents 
    };

    agency::experimental::array<T, mgpu::max(num_agents, 2 * num_participating_agents)> storage_;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
    warp_reducing_barrier<T, num_participating_agents> warp_barrier_;
#endif
};


template<typename launch_arg_t = mgpu::empty_t, typename input_it,  typename output_it, typename op_t>
void my_reduce(input_it input, int count, output_it reduction, op_t op, mgpu::context_t& context)
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

  auto k = [=] __device__ (int agent_idx, int block_idx)
  {
    typedef typename launch_t::sm_ptx params_t;

    constexpr int num_threads = params_t::nt;
    constexpr int grainsize = params_t::vt;
    constexpr int tile_size = num_threads * grainsize;

    // Load the data for the first tile for each cta.
    range_t tile = get_tile(block_idx, tile_size, count);

    // stride through the input and compute a partial sum per thread
    span<T> our_span(input + tile.begin, input + tile.end);
    auto my_values = strided(drop(our_span, agent_idx), size_t(num_threads));

    // we don't have an initializer for the agent's sum, so use uninitialized_reduce
    auto partial_sum = uninitialized_reduce(bound<grainsize>(), my_values, op);

    // Reduce to a scalar per CTA.
    int num_partials = min(tile.count(), (int)num_threads);

    __shared__ reducing_barrier<T, num_threads> barrier;
    auto result = barrier.reduce_and_wait_and_elect(agent_idx, partial_sum, num_partials, op);

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

