#pragma once

#include <utility>
#include <type_traits>
#include <agency/experimental/view.hpp>
#include "../algorithm.hpp"

template<class ExecutionPolicy, class Range, class T, class BinaryOperator>
__host__ __device__
enable_if_sequential_t<typename std::decay<ExecutionPolicy>::type, T>
  accumulate(ExecutionPolicy policy, Range&& rng, T init, BinaryOperator binary_op)
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
  accumulate_nonempty(ExecutionPolicy policy, Range&& rng, BinaryOperator binary_op)
{
  return accumulate(policy, agency::experimental::drop(std::forward<Range>(rng), 1), std::forward<Range>(rng)[0], binary_op);
}

