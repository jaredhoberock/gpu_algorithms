#pragma once

#include <agency/agency.hpp>
#include <agency/execution/executor/experimental/unrolling_executor.hpp>
#include <utility>
#include <type_traits>

template<size_t factor>
struct unrolling_executor
{
  using execution_category = agency::sequenced_execution_tag;

  __AGENCY_ANNOTATION
  constexpr std::size_t unit_shape() const
  {
    return factor;
  }

  template<class Function, class ResultFactory, class SharedFactory>
  __AGENCY_ANNOTATION
  typename std::result_of<ResultFactory()>::type
    bulk_sync_execute(Function f, std::size_t n, ResultFactory result_factory, SharedFactory shared_factory)
  {
    auto result = result_factory();
    auto shared_arg = shared_factory();

    if(n == factor)
    {
      #pragma unroll
      for(size_t idx = 0; idx < factor; ++idx)
      {
        f(idx, result, shared_arg);
      }
    }
    else if(n < factor)
    {
      #pragma unroll
      for(size_t idx = 0; idx < factor; ++idx)
      {
        if(idx < n)
        {
          f(idx, result, shared_arg);
        }
      }
    }
    else
    {
      for(size_t idx = 0; idx < n; )
      {
        #pragma unroll
        for(size_t i = 0; i < factor; ++i, ++idx)
        {
          if(idx < n)
          {
            f(idx, result, shared_arg);
          }
        }
      }
    }

    return std::move(result);
  }
};


template<size_t factor_>
class unrolling_execution_policy : public agency::basic_execution_policy<
  agency::sequenced_agent,
  agency::experimental::unrolling_executor<factor_>,
  unrolling_execution_policy<factor_>
>
{
  private:
    using super_t = agency::basic_execution_policy<
      agency::sequenced_agent,
      agency::experimental::unrolling_executor<factor_>,
      unrolling_execution_policy<factor_>
    >;

  public:
    static constexpr size_t factor = factor_;

    using super_t::basic_execution_policy;
};

template<size_t factor>
__AGENCY_ANNOTATION
unrolling_execution_policy<factor> unroll()
{
  return unrolling_execution_policy<factor>{};
}

