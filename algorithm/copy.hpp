#pragma once

#include <cstddef>
#include <agency/execution_agent.hpp>
#include <agency/experimental/span.hpp>
#include <agency/experimental/chunk.hpp>
#include "../algorithm.hpp"
#include "../bound.hpp"

template<std::size_t bound, class Range1, class Range2>
__device__
void sequential_bounded_copy(Range1&& in, Range2&& out)
{
  for(std::size_t i = 0; i < bound; ++i)
  {
    if(i < in.size())
    {
      out[i] = in[i];
    }
  }
}


template<std::size_t group_size, std::size_t grain_size, class Range1, class Range2>
__device__
void bounded_copy(agency::experimental::static_concurrent_agent<group_size, grain_size>& self, const Range1& in, Range2&& out)
{
  using namespace agency::experimental;

  // create strided views of each range
  auto view_of_this_agents_input  = stride(drop(in, self.rank()), size_t(group_size));

  auto view_of_this_agents_output = stride(drop(out, self.rank()), size_t(group_size));

  // each agent copies sequentially at most grain_size elements
  copy(bound<grain_size>(), view_of_this_agents_input, view_of_this_agents_output.begin());

  // wait for all agents to finish copying before returning
  self.wait();
}

