#pragma once

#include <agency/agency.hpp>

template<size_t bound>
struct bounded_executor
{
  using execution_category = agency::sequential_execution_tag;

  __host__ __device__
  constexpr std::size_t shape() const
  {
    return bound;
  }

  template<class Function, class Factory>
  __host__ __device__
  void execute(Function f, std::size_t n, Function shared_factory)
  {
    auto shared_parm = shared_factory();

    for(size_t idx = 0; idx < bound; ++idx)
    {
      if(idx < n)
      {
        f(idx, shared_parm);
      }
    }
  }

  template<class Function>
  __host__ __device__
  void execute(Function f, std::size_t n)
  {
    for(size_t idx = 0; idx < bound; ++idx)
    {
      if(idx < n)
      {
        f(idx);
      }
    }
  }
};


template<size_t bound_>
class bounded_execution_policy : public agency::detail::basic_execution_policy<
  agency::sequential_agent,
  bounded_executor<bound_>,
  agency::sequential_execution_tag,
  bounded_execution_policy<bound_>
>
{
  private:
    using super_t = agency::detail::basic_execution_policy<
      agency::sequential_agent,
      bounded_executor<bound_>,
      agency::sequential_execution_tag,
      bounded_execution_policy<bound_>
    >;

  public:
    static constexpr size_t bound = bound_;

    using super_t::basic_execution_policy;
};

template<size_t b>
__host__ __device__
bounded_execution_policy<b> bound()
{
  return bounded_execution_policy<b>{};
}

