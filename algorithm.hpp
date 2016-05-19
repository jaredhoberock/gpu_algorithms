#pragma once

#include <agency/agency.hpp>
#include <agency/experimental/view.hpp>
#include <type_traits>

template<class ExecutionPolicy, class Result = void>
struct enable_if_sequential
  : std::enable_if<
      std::is_same<
        agency::sequential_execution_tag,
        typename ExecutionPolicy::execution_category
      >::value,
      Result
    >
{};

template<class ExecutionPolicy, class Result = void>
using enable_if_sequential_t = typename enable_if_sequential<ExecutionPolicy,Result>::type;


template<size_t bound, class Range, class Function>
__host__ __device__
agency::experimental::detail::decay_range_iterator_t<Range> for_each(bounded_execution_policy<bound> policy, Range&& rng, Function f)
{
  auto iter = rng.begin();

  bounded_executor<bound> exec;

  exec.execute([&](size_t i)
  {
    if(iter != rng.end())
    {
      std::forward<Function>(f)(*iter);
      ++iter;
    }
  },
  bound);

  return rng.begin();
}


template<class Range, class Function>
__host__ __device__
agency::experimental::detail::decay_range_iterator_t<Range>
  for_each(agency::sequential_execution_policy, Range&& rng, Function f)
{
  for(auto i = rng.begin(); i != rng.end(); ++i)
  {
    std::forward<Function>(f)(*i);
  }

  return rng.begin();
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
OutputIterator copy(agency::sequential_execution_policy, Range&& rng, OutputIterator out)
{
  for(auto iter = rng.begin(); iter != rng.end(); ++iter, ++out)
  {
    *out = *iter;
  }

  return out;
}


template<class ExecutionPolicy, class Range, class T, class BinaryOperator>
__host__ __device__
enable_if_sequential_t<typename std::decay<ExecutionPolicy>::type, T>
  reduce(ExecutionPolicy policy, Range&& rng, T init, BinaryOperator binary_op)
{
  using value_type = typename agency::experimental::detail::range_value_t<typename std::decay<Range>::type>;

  for_each(policy, std::forward<Range>(rng), [&](value_type& value)
  {
    init = binary_op(init, value);
  });

  return init;
}


template<class ExecutionPolicy, class Range, class BinaryOperator>
__host__ __device__
agency::experimental::detail::range_value_t<typename std::decay<Range>::type>
  reduce_nonempty(ExecutionPolicy policy, Range&& rng, BinaryOperator binary_op)
{
  return reduce(policy, agency::experimental::drop(std::forward<Range>(rng), 1), std::forward<Range>(rng)[0], binary_op);
}


template<class ExecutionPolicy, class Range, class BinaryOperator>
__host__ __device__
agency::experimental::optional<agency::experimental::detail::range_value_t<typename std::decay<Range>::type>>
  uninitialized_reduce(ExecutionPolicy policy, Range&& rng, BinaryOperator binary_op)
{
  if(!std::forward<Range>(rng).empty())
  {
    return ::reduce_nonempty(policy, std::forward<Range>(rng), binary_op);
  }

  return agency::experimental::nullopt;
}

