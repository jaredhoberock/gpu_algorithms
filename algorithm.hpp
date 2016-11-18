#pragma once

#include <agency/agency.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <type_traits>
#include "collective_ptr.hpp"
#include "bound.hpp"
#include "unroll.hpp"
#include "reducing_barrier.hpp"

template<class ExecutionPolicy, class Result = void>
struct enable_if_sequenced
  : std::enable_if<
      std::is_same<
        agency::sequenced_execution_tag,
        typename agency::execution_agent_traits<typename ExecutionPolicy::execution_agent_type>::execution_category
      >::value,
      Result
    >
{};

template<class ExecutionPolicy, class Result = void>
using enable_if_sequenced_t = typename enable_if_sequenced<ExecutionPolicy,Result>::type;


template<size_t bound, class Range, class Function>
__host__ __device__
void for_each(bounded_execution_policy<bound> policy, Range&& rng, Function f)
{
  bounded_executor<bound> exec;

  exec.bulk_sync_execute([&](size_t i, int, int)
  {
    std::forward<Function>(f)(rng[i]);
  },
  rng.size(),
  []{ return 0; },
  []{ return 0; }
  );
}


template<size_t factor, class Range, class Function>
__host__ __device__
void for_each(unrolling_execution_policy<factor> policy, Range&& rng, Function f)
{
  unrolling_executor<factor> exec;

  exec.bulk_sync_execute([&](size_t i, int, int)
  {
    std::forward<Function>(f)(rng[i]);
  },
  rng.size(),
  []{ return 0; },
  []{ return 0; }
  );
}


template<class Range, class Function>
__host__ __device__
void for_each(agency::sequenced_execution_policy, Range&& rng, Function f)
{
  for(auto i = rng.begin(); i != rng.end(); ++i)
  {
    std::forward<Function>(f)(*i);
  }
}


// XXX implement copy with for_each

template<size_t bound, class Range, class OutputIterator>
__host__ __device__
OutputIterator copy(bounded_execution_policy<bound> policy, Range&& rng, OutputIterator out)
{
  auto in = rng.begin();
  auto result = out;

  for(size_t i = 0; i < bound; ++i)
  {
    if(in != rng.end())
    {
      out[i] = rng[i];
      ++in;
      ++result;
    }
  }

  return result;
}


template<class Range, class OutputIterator>
__host__ __device__
OutputIterator copy(agency::sequenced_execution_policy, Range&& rng, OutputIterator out)
{
  for(auto iter = rng.begin(); iter != rng.end(); ++iter, ++out)
  {
    *out = *iter;
  }

  return out;
}


template<class ExecutionPolicy, class Range, class T, class BinaryOperator>
__host__ __device__
enable_if_sequenced_t<typename std::decay<ExecutionPolicy>::type, T>
  reduce(ExecutionPolicy policy, Range&& rng, T init, BinaryOperator binary_op)
{
  using value_type = typename agency::experimental::range_value_t<Range>;

  for_each(policy, std::forward<Range>(rng), [&](value_type& value)
  {
    init = binary_op(init, value);
  });

  return init;
}


template<class ExecutionPolicy, class Range, class BinaryOperator>
__host__ __device__
agency::experimental::range_value_t<Range>
  reduce_nonempty(ExecutionPolicy policy, Range&& rng, BinaryOperator binary_op)
{
  return reduce(policy, agency::experimental::drop(std::forward<Range>(rng), 1), std::forward<Range>(rng)[0], binary_op);
}


template<class ExecutionPolicy, class Range, class BinaryOperator>
__host__ __device__
agency::experimental::optional<agency::experimental::range_value_t<Range>>
  uninitialized_reduce(ExecutionPolicy policy, Range&& rng, BinaryOperator binary_op)
{
  if(!std::forward<Range>(rng).empty())
  {
    return ::reduce_nonempty(policy, std::forward<Range>(rng), binary_op);
  }

  return agency::experimental::nullopt;
}


template<std::size_t group_size, std::size_t grain_size, class Range, class BinaryOperator>
__host__ __device__
agency::experimental::range_value_t<Range>
  uninitialized_reduce(agency::experimental::static_concurrent_agent<group_size, grain_size>& self,
                       Range&& rng,
                       BinaryOperator binary_op)
{
  auto agent_rank = self.rank();

  // each agent strides through its group's chunk of the input...
  auto my_values = stride(drop(rng, agent_rank), size_t(group_size));
  
  // ...and sequentially computes a partial sum
  auto partial_sum = uninitialized_reduce(unroll<grain_size>(), my_values, binary_op);
  
  // the entire group cooperatively reduces the partial sums
  int num_partials = rng.size() < group_size ? rng.size() : group_size;
  
  using T = agency::experimental::range_value_t<Range>;
  using reduce_t = reducing_barrier<T,group_size>;
  auto reducer_ptr = make_collective<reduce_t>(self);
  return reducer_ptr->reduce_and_wait(self, partial_sum, num_partials, binary_op);
}


template<std::size_t group_size, std::size_t grain_size, class Range, class BinaryOperator>
__host__ __device__
agency::experimental::optional<agency::experimental::range_value_t<Range>>
  uninitialized_reduce_and_elect(agency::experimental::static_concurrent_agent<group_size, grain_size>& self,
                                 Range&& rng,
                                 BinaryOperator binary_op)
{
  auto agent_rank = self.rank();

  // each agent strides through its group's chunk of the input...
  auto my_values = stride(drop(rng, agent_rank), size_t(group_size));
  
  // ...and sequentially computes a partial sum
  auto partial_sum = uninitialized_reduce(bound<grain_size>(), my_values, binary_op);
  
  // the entire group cooperatively reduces the partial sums
  int num_partials = rng.size() < group_size ? rng.size() : group_size;
  
  using T = agency::experimental::range_value_t<Range>;
  using reduce_t = reducing_barrier<T,group_size>;
  auto reducer_ptr = make_collective<reduce_t>(self);
  return reducer_ptr->reduce_and_wait_and_elect(self, partial_sum, num_partials, binary_op);
}

