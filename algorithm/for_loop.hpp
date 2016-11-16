#pragma once

#include <agency/detail/config.hpp>
#include <utility>
#include <cstddef>
#include <limits>
#include "../algorithm.hpp"

namespace detail
{


// XXX we should relax the type of first and last
template<std::size_t first, std::size_t last>
struct static_for_loop_impl
{
  template<class Function>
  __AGENCY_ANNOTATION
  static void invoke(Function&& f)
  {
    std::forward<Function>(f)(first);

    static_for_loop_impl<first+1,last>::invoke(std::forward<Function>(f));
  }
};


template<std::size_t first>
struct static_for_loop_impl<first,first>
{
  template<class Function>
  __AGENCY_ANNOTATION
  static void invoke(Function&&)
  {
  }
};


// this one doesn't seem to unroll like static_for_loop_impl
template<class Function>
__AGENCY_ANNOTATION
constexpr int constexpr_static_for_loop_impl(int first, int last, Function&& f)
{
  return (first == last) ? 0 : (std::forward<Function>(f)(first), new_static_for_loop(first+1,last,std::forward<Function>(f)));
}


} // end detail


template<class Size, Size n, class Function>
__AGENCY_ANNOTATION
void static_for_loop(Function&& f)
{
  detail::static_for_loop_impl<0,n>::invoke(std::forward<Function>(f));
}


template<std::size_t n, class Function>
__AGENCY_ANNOTATION
void static_for_loop(Function&& f)
{
  static_for_loop<std::size_t,n>(std::forward<Function>(f));
}


template<class Size, Size bound, class Function>
__AGENCY_ANNOTATION
void bounded_for_loop(Size n, Function&& f)
{
  if(bound == n)
  {
    static_for_loop<Size,bound>(std::forward<Function>(f));
  }
  else
  {
    static_for_loop<Size,bound>([&](Size idx)
    {
      if(idx < n)
      {
        std::forward<Function>(f)(idx);
      }
    });
  }
}


template<std::size_t bound, class Function>
__AGENCY_ANNOTATION
void bounded_for_loop(std::size_t n, Function&& f)
{
  bounded_for_loop<std::size_t,bound>(n, std::forward<Function>(f));
}


template<class Size, class Function>
__AGENCY_ANNOTATION
typename std::enable_if<
  (std::numeric_limits<Size>::max() <= 32)
>::type
  for_loop(Size n, Function&& f)
{
  constexpr Size bound = std::numeric_limits<Size>::max();

  bounded_for_loop<Size,bound>(n, std::forward<Function>(f));
}


template<class Size, class Function>
__AGENCY_ANNOTATION
typename std::enable_if<
  (std::numeric_limits<Size>::max() > 32)
>::type
  for_loop(Size n, Function&& f)
{
  for(Size i = 0; i < n; ++i)
  {
    std::forward<Function>(f)(i);
  }
}


template<size_t bound, class Size, class Function>
__AGENCY_ANNOTATION
void for_loop(bounded_execution_policy<bound> policy, Size n, Function f)
{
  bounded_executor<bound> exec;

  exec.execute([&](size_t i)
  {
    std::forward<Function>(f)(i);
  },
  n);
}


template<class Size, class Function>
__AGENCY_ANNOTATION
void for_loop(agency::sequenced_execution_policy, Size n, Function f)
{
  for(Size i = 0; i < n; ++i)
  {
    std::forward<Function>(f)(i);
  }
}

