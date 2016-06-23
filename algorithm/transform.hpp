#pragma once

#include "for_loop.hpp"

// XXX receiving UnaryOperation as a reference seems important for conserving registers
//     in sequential __device__ code
template<class Range1, class Range2, class UnaryOperation>
__AGENCY_ANNOTATION
void transform(const Range1& in, Range2&& out, UnaryOperation unary_op)
{
  for_loop(in.size(), [&](int idx)
  {
    out[idx] = unary_op(in[idx]);
  });
}

