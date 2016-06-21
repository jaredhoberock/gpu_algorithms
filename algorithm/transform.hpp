#pragma once

#include "for_loop.hpp"

// XXX receiving UnaryOperation as a reference seems important for conserving registers
//     in sequential __device__ code
template<size_t bound, class Range1, class Range2, class UnaryOperation>
__AGENCY_ANNOTATION
void bounded_transform(const Range1& in, Range2&& out, UnaryOperation unary_op)
{
  bounded_for_loop<bound>(in.size(), [&](int idx)
  {
    out[idx] = unary_op(in[idx]);
  });
}

