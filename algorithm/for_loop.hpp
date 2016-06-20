#pragma once

#include <utility>
#include "../algorithm.hpp"

namespace detail
{


template<size_t first, size_t last>
struct static_for_loop_impl
{
  template<class Function>
  __host__ __device__
  static void invoke(Function&& f)
  {
    std::forward<Function>(f)(first);

    static_for_loop_impl<first+1,last>::invoke(std::forward<Function>(f));
  }
};


template<size_t first>
struct static_for_loop_impl<first,first>
{
  template<class Function>
  __host__ __device__
  static void invoke(Function&&)
  {
  }
};


} // end detail


template<size_t n, class Function>
__host__ __device__
void static_for_loop(Function&& f)
{
  detail::static_for_loop_impl<0,n>::invoke(std::forward<Function>(f));
}


template<size_t bound, class Size, class Function>
__host__ __device__
void bounded_for_loop(Size n, Function&& f)
{
  if(bound == n)
  {
    static_for_loop<bound>(std::forward<Function>(f));
  }
  else
  {
    static_for_loop<bound>([&](Size idx)
    {
      if(idx < n)
      {
        std::forward<Function>(f)(idx);
      }
    });
  }
}


template<size_t bound, class Size, class Function>
__host__ __device__
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
__host__ __device__
void for_loop(agency::sequential_execution_policy, Size n, Function f)
{
  for(Size i = 0; i < n; ++i)
  {
    std::forward<Function>(f)(i);
  }
}

