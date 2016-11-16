#pragma once

#include <cstddef>
#include <agency/agency.hpp>
#include <agency/experimental/ranges.hpp>
#include "../algorithm.hpp"
#include "for_loop.hpp"
#include "../bound.hpp"

template<std::size_t bound, class Range1, class Range2>
__device__
void sequential_bounded_copy(Range1&& in, Range2&& out)
{
  bounded_for_loop<bound>(in.size(), [&](int i)
  {
    out[i] = in[i];
  });
}


namespace detail
{

template<class Tuple>
__AGENCY_ANNOTATION
void assign_first_to_second(const Tuple& t)
{
  agency::detail::get<1>(t) = agency::detail::get<0>(t);
}

}


template<std::size_t group_size, std::size_t grain_size, class Range1, class Range2>
__device__
void bounded_copy(agency::experimental::static_concurrent_agent<group_size, grain_size>& self, const Range1& in, Range2&& out)
{
  // XXX this implementation may be less efficient than the one below
  //     because it computes strided indices twice
  using namespace agency::experimental;

  // create strided views of each range
  auto view_of_this_agents_input  = stride(drop(in, self.rank()), size_t(group_size));

  auto view_of_this_agents_output = stride(drop(out, self.rank()), size_t(group_size));

  // each agent copies sequentially at most grain_size elements
  copy(bound<grain_size>(), view_of_this_agents_input, view_of_this_agents_output.begin());

  // wait for all agents to finish copying before returning
  self.wait();
}

// XXX eliminate this and replace the above's implementation with this one
template<std::size_t group_size, std::size_t grain_size, class Range1, class Range2>
__device__
void bounded_copy(int tid, const Range1& in, Range2&& out)
{
  auto in_and_out = zip(in, out);
  auto view_of_this_agents_range = stride(drop(in_and_out, tid), group_size);

  // XXX if we were able to express this as a cyclic_split instead of a stride(),
  //     we could probably compute the size of the agent's range without a divide
  //     because we'd have more information when computing the size()

  if(in_and_out.size() == group_size * grain_size)
  {
    static_for_loop<grain_size>([&](int i)
    {
      detail::assign_first_to_second(view_of_this_agents_range[i]);
    });
  }
  else
  {
    bounded_for_loop<std::size_t,grain_size>(view_of_this_agents_range.size(), [&](int i)
    {
      detail::assign_first_to_second(view_of_this_agents_range[i]);
    });
  }

  __syncthreads();
}


template<class Range, std::size_t N>
struct range_size_is_always_less_than_or_equal_to
{
  using size_type = agency::experimental::range_size_t<Range>;

  static const bool value = std::numeric_limits<size_type>::max() <= N;
};


template<class Range, std::size_t N, class Result = void>
struct enable_if_range_size_is_always_less_than_or_equal_to
  : std::enable_if<
      range_size_is_always_less_than_or_equal_to<Range,N>::value,
      Result
    >
{};


// this overload of copy copies a "tile" of elements, which is a range of elements whose size
// is less than or equal to the agent's group size times its grain size
template<std::size_t group_size, std::size_t grain_size, class Range1, class Range2>
__device__
typename enable_if_range_size_is_always_less_than_or_equal_to<Range1,group_size * grain_size>::type
  collective_copy(agency::experimental::static_concurrent_agent<group_size,grain_size>& self,
                  const Range1& in,
                  Range2&& out)
{
  auto in_and_out = zip(in, out);
  auto view_of_this_agents_range = stride(drop(in_and_out, self.rank()), group_size);

  // XXX if we were able to express this as a cyclic_split instead of a stride(),
  //     we could probably compute the size of the agent's range without a divide
  //     because we'd have more information when computing the size()

  if(in_and_out.size() == group_size * grain_size)
  {
    static_for_loop<grain_size>([&](int i)
    {
      detail::assign_first_to_second(view_of_this_agents_range[i]);
    });
  }
  else
  {
    bounded_for_loop<std::size_t,grain_size>(view_of_this_agents_range.size(), [&](int i)
    {
      detail::assign_first_to_second(view_of_this_agents_range[i]);
    });
  }

  self.wait();
}

