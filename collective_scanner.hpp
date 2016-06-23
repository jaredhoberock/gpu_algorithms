#pragma once

#include <agency/experimental/optional.hpp>

template<class T, int group_size>
class collective_scanner
{
  private:
    T storage_[2 * group_size];

  public:
    template<class ConcurrentAgent, class BinaryOperation>
    __device__
    T inplace_exclusive_scan(ConcurrentAgent& self, agency::experimental::optional<T>& summand, int count, T init, BinaryOperation binary_op)
    {
      int rank = self.rank();

      // the first agent accumulates init into its summand
      if(rank == 0 && count > 0)
      {
        summand = binary_op(init, *summand);
      }

      // all agents store their summand to temporary storage
      if(rank < count)
      {
        storage_[rank] = *summand;
      }

      self.wait();

      // double buffer to eliminate one barrier in the loop below 
      int first = 0;

      for(int offset = 1; offset < count; offset += offset)
      {
        if(rank >= offset)
        {
          *summand = binary_op(storage_[first + rank - offset], *summand);
        }

        first = group_size - first;

        storage_[first + rank] = *summand;

        self.wait();
      }

      T carry_out = count > 0 ? storage_[first + count - 1] : init;

      if(rank < count)
      {
        if(rank == 0)
        {
          *summand = init;
        }
        else
        {
          *summand = storage_[first + rank - 1];
        }
      }

      self.wait();

      return carry_out;
    }
};

