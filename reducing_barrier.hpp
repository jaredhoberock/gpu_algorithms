#pragma once

#include <agency/experimental/optional.hpp>
#include <agency/cuda/execution_agent.hpp>
#include <cstddef>


template<class T>
__host__ __device__
constexpr T minimum(const T& a, const T& b)
{
  return (b < a) ? b : a;
}

template<class T>
__host__ __device__
constexpr T maximum(const T& a, const T& b)
{
  return (a < b) ? b : a;
}


template<class Integer>
__host__ __device__
constexpr bool is_pow2(Integer x)
{
  return 0 == (x & (x - 1));
}


template<class Integer>
__host__ __device__
constexpr Integer div_up(Integer numerator, Integer denominator)
{
  return (numerator + denominator - Integer(1)) / denominator;
}


template<class Integer>
__host__ __device__
constexpr Integer log2(Integer x)
{
  return (x > 1) ? (1 + log2(x/2)) : 0;
}


// XXX seems like this is really only valid for POD
// XXX if we destroy x and properly construct the result, how correct does that make it?
template<class T>
__device__
T shuffle_down(const T& x, int offset, int width)
{ 
  constexpr std::size_t num_words = div_up(sizeof(T), sizeof(int));

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
  constexpr std::size_t num_words = div_up(sizeof(T), sizeof(int));

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
  static_assert(num_threads <= 32 && is_pow2(num_threads), "shfl_reduce_t must operate on a pow2 number of threads <= CUDA warp size (32)");

  template<typename BinaryOperation>
  __device__
  agency::experimental::optional<T> reduce_and_wait_and_elect(int lane, agency::experimental::optional<T> x, int count, BinaryOperation binary_op) const
  {
    if(count == num_threads)
    { 
      for(int pass = 0; pass < log2(num_threads); ++pass)
      {
        int offset = 1 << pass;
        auto y = shuffle_down(*x, offset, num_threads);
        x = binary_op(*x, y);
      }
    }
    else
    {
      for(int pass = 0; pass < log2(num_threads); ++pass)
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
    static_assert(0 == num_agents % 32, "num_agents must be a multiple of warp_size (32)");
   
    __device__
    reducing_barrier() = default;

    reducing_barrier(const reducing_barrier&) = delete;

    reducing_barrier(reducing_barrier&&) = delete;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
    template<class ConcurrentAgent, class BinaryOperation>
    __device__
    agency::experimental::optional<T> reduce_and_wait_and_elect(ConcurrentAgent& self, const agency::experimental::optional<T>& value, int count, BinaryOperation binary_op)
    {
      auto agent_rank = self.rank();
      auto partial_sum = value;

      // store partial sum to storage
      if(agent_rank < count)
      {
        storage_[agent_rank] = *partial_sum;
      }

      using namespace agency::experimental;
      auto partial_sums = span<T>(storage_.data(), count);
      self.wait();

      if(agent_rank < num_participating_agents)
      {
        // stride through the input and compute a partial sum per agent
        auto my_partial_sums = strided(drop(partial_sums, agent_rank), (int)num_participating_agents);

        partial_sum = ::uninitialized_reduce(bound<num_sequential_sums_per_agent>(), my_partial_sums, binary_op);

        // reduce across the warp
        partial_sum = warp_barrier_.reduce_and_wait_and_elect(agent_rank, partial_sum, minimum(count, (int)num_participating_agents), binary_op);
      }
      self.wait();

      return agent_rank == 0 ? partial_sum : agency::experimental::nullopt;
    }

#else

    template<class ConcurrentAgent, class BinaryOperation>
    __device__
    agency::experimental::optional<T> reduce_and_wait_and_elect(ConcurrentAgent& self, const agency::experimental::optional<T>& value, int count, BinaryOperation binary_op)
    {
      auto agent_rank = self.rank();
      auto partial_sum = value;

      // store partial sum to storage
      if(agent_rank < count)
      {
        storage_[agent_rank] = *partial_sum;
      }

      using namespace agency::experimental;
      auto partial_sums = span<T>(storage_.data(), count);
      self.wait();

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
      self.wait();

      int count2 = minimum(count, int(num_participating_agents));
      int first = (1 & num_passes) ? num_participating_agents : 0;
      if(agent_rank < num_participating_agents && partial_sum)
      {
        storage_[first + agent_rank] = *partial_sum;
      }
      self.wait();


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
        self.wait();
      }

      return agent_rank == 0 ? partial_sum : agency::experimental::nullopt;
    }

#endif

    template<class ConcurrentAgent, class BinaryOperation>
    __device__
    T reduce_and_wait(ConcurrentAgent& self, const agency::experimental::optional<T>& value, int count, BinaryOperation binary_op) const
    {
      auto result = reduce_and_wait_and_elect(self, value, count, binary_op);

      // XXX we're using inside knowledge that reduce_and_elect() always elects agent_rank == 0
      if(self.rank() == 0)
      {
        storage_[0] = *result;
      }

      self.wait();

      return storage_[0];
    }

  private:
    // XXX these should just be constexpr members, but nvcc crashes when i do that
    enum
    { 
      num_participating_agents = minimum(num_agents, (int)32), 
      num_passes = log2(num_participating_agents),
      num_sequential_sums_per_agent = num_agents / num_participating_agents 
    };

    agency::experimental::array<T, maximum(num_agents, 2 * num_participating_agents)> storage_;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
    warp_reducing_barrier<T, num_participating_agents> warp_barrier_;
#endif
};

