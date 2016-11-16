#pragma once

#include <agency/agency.hpp>
#include <utility>
#include <type_traits>

template<size_t bound>
struct bounded_executor
{
  using execution_category = agency::sequenced_execution_tag;

  __AGENCY_ANNOTATION
  constexpr std::size_t unit_shape() const
  {
    return bound;
  }

  template<class Function, class ResultFactory, class SharedFactory>
  __AGENCY_ANNOTATION
  typename std::result_of<ResultFactory()>::type
    bulk_sync_execute(Function f, std::size_t n, ResultFactory result_factory, SharedFactory shared_factory)
  {
    auto result = result_factory();
    auto shared_arg = shared_factory();

    for(size_t idx = 0; idx < bound; ++idx)
    {
      if(idx < n)
      {
        f(idx, result, shared_arg);
      }
    }

    return std::move(result);
  }
};


template<size_t bound_>
class bounded_execution_policy : public agency::basic_execution_policy<
  agency::sequenced_agent,
  bounded_executor<bound_>,
  bounded_execution_policy<bound_>
>
{
  private:
    using super_t = agency::basic_execution_policy<
      agency::sequenced_agent,
      bounded_executor<bound_>,
      bounded_execution_policy<bound_>
    >;

  public:
    static constexpr size_t bound = bound_;

    using super_t::basic_execution_policy;
};

template<size_t b>
__AGENCY_ANNOTATION
bounded_execution_policy<b> bound()
{
  return bounded_execution_policy<b>{};
}

