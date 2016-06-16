#pragma once

#include "../bound.hpp"
#include "../algorithm.hpp"

template<size_t bound, class Range1, class Range2, class T, class BinaryOperation>
__host__ __device__
void exclusive_scan(bounded_execution_policy<bound> policy, Range1&& in, Range2&& out, T init, BinaryOperation binary_op)
{
  auto input_size = in.size();

  // XXX implement with for_loop()? 

  if(bound <= input_size)
  {
    for(size_t i = 0; i < bound; ++i)
    {
      // the temporary value allows in-situ scan
      auto tmp = in[i];

      out[i] = init;
      init = binary_op(init, tmp);
    }
  }
  else
  {
    for(size_t i = 0; i < bound; ++i)
    {
      if(i < input_size)
      {
        // the temporary value allows in-situ scan
        auto tmp = in[i];

        out[i] = init;
        init = binary_op(init, tmp);
      }
    }
  }
}

