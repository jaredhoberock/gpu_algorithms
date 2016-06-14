#pragma once

#include "../bound.hpp"
#include "../algorithm.hpp"

template<size_t bound, class Range, class BinaryOperation, class T>
__host__ __device__
void inclusive_scan(bounded_execution_policy<bound> policy, Range&& in, Range&& out, BinaryOperation binary_op, T init)
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

template<size_t bound, class Range, class BinaryOperation>
__host__ __device__
void inclusive_scan(bounded_execution_policy<bound> policy, Range&& in, Range&& out, BinaryOperation binary_op)
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

