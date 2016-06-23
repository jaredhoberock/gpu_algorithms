#pragma once

#include "../bound.hpp"
#include "../algorithm.hpp"
#include "for_loop.hpp"

template<size_t bound, class Range1, class Range2, class BinaryOperation, class T>
__AGENCY_ANNOTATION
void inclusive_scan(bounded_execution_policy<bound> policy, Range1&& in, Range2&& out, BinaryOperation binary_op, T init)
{
  auto input_size = in.size();

  // XXX implement with for_loop()? 

  if(bound <= input_size)
  {
    for(size_t i = 0; i < bound; ++i)
    {
      if(i == 0)
      {
        out[i] = binary_op(init, in[i]);
      }
      else
      {
        out[i] = binary_op(out[i - 1], in[i]);
      }
    }
  }
  else
  {
    for(size_t i = 0; i < bound; ++i)
    {
      if(i < input_size)
      {
        if(i == 0)
        {
          out[i] = binary_op(init, in[i]);
        }
        else
        {
          out[i] = binary_op(out[i - 1], in[i]);
        }
      }
    }
  }
}

template<size_t bound, class Range1, class Range2, class BinaryOperation>
__AGENCY_ANNOTATION
void inclusive_scan(bounded_execution_policy<bound> policy, Range1&& in, Range2&& out, BinaryOperation binary_op)
{
  auto input_size = in.size();

  // XXX implement with for_loop()? 

  if(bound <= input_size)
  {
    for(size_t i = 0; i < bound; ++i)
    {
      if(i == 0)
      {
        out[i] = in[i];
      }
      else
      {
        out[i] = binary_op(in[i], out[i - 1]);
      }
    }
  }
  else
  {
    for(size_t i = 0; i < bound; ++i)
    {
      if(i < input_size)
      {
        if(i == 0)
        {
          out[i] = in[i];
        }
        else
        {
          out[i] = binary_op(in[i], out[i - 1]);
        }
      }
    }
  }
}


template<class Range1, class Range2, class BinaryOperation>
__AGENCY_ANNOTATION
void inclusive_scan(const Range1& in, Range2&& out, BinaryOperation binary_op)
{
  for_loop(in.size(), [&](int i)
  {
    if(i == 0)
    {
      out[i] = in[i];
    }
    else
    {
      out[i] = binary_op(in[i], out[i - 1]);
    }
  });
}


// XXX eliminate this
template<size_t bound, class Range1, class Range2, class BinaryOperation>
__AGENCY_ANNOTATION
void inclusive_scan(const Range1& in, Range2&& out, BinaryOperation binary_op)
{
  bounded_for_loop<bound>(in.size(), [&](int i)
  {
    if(i == 0)
    {
      out[i] = in[i];
    }
    else
    {
      out[i] = binary_op(in[i], out[i - 1]);
    }
  });
}


template<class Range1, class Range2, class BinaryOperation, class T>
__AGENCY_ANNOTATION
void inclusive_scan(const Range1& in, Range2&& out, BinaryOperation binary_op, T init)
{
  for_loop(in.size(), [&](int i)
  {
    if(i == 0)
    {
      out[i] = binary_op(init, in[i]);
    }
    else
    {
      out[i] = binary_op(in[i], out[i - 1]);
    }
  });
}


// XXX eliminate this
template<size_t bound, class Range1, class Range2, class BinaryOperation, class T>
__AGENCY_ANNOTATION
void inclusive_scan(const Range1& in, Range2&& out, BinaryOperation binary_op, T init)
{
  bounded_for_loop<bound>(in.size(), [&](int i)
  {
    if(i == 0)
    {
      out[i] = binary_op(init, in[i]);
    }
    else
    {
      out[i] = binary_op(in[i], out[i - 1]);
    }
  });
}

