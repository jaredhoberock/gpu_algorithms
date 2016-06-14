#pragma once

#include <moderngpu/loadstore.hxx>
#include <moderngpu/intrinsics.hxx>
#include <agency/experimental/array.hpp>
#include "algorithm/inclusive_scan.hpp"
#include "algorithm/exclusive_scan.hpp"
#include "algorithm/accumulate.hpp"

using namespace mgpu;


template<typename type_t, int vt = 0, bool is_array = (vt > 0)>
struct my_scan_result_t {
  type_t scan;
  type_t reduction;
};

template<typename type_t, int vt>
struct my_scan_result_t<type_t, vt, true> {
  agency::experimental::array<type_t, vt> scan;
  type_t reduction;
};

////////////////////////////////////////////////////////////////////////////////

template<int nt, typename type_t>
struct my_cta_scan_t
{
  enum { num_warps = nt / warp_size, capacity = nt + num_warps };

  union storage_t {
    type_t data[2 * nt];
    struct { type_t threads[nt], warps[num_warps]; };
  };

//#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300  
//
//  //////////////////////////////////////////////////////////////////////////////
//  // Optimized CTA scan code that uses warp shfl intrinsics.
//  // Shfl is used for all data types, not just 4-byte built-in types, however
//  // those have accelerated plus, maximum and minimum operators.
//
//  template<typename op_t = plus_t<type_t> >
//  __device__
//  my_scan_result_t<type_t>
//    scan(int tid, type_t x, storage_t& storage, int count = nt, op_t op = op_t(), type_t init = type_t(), mgpu::scan_type_t type = mgpu::scan_type_exc) const
//  {
//    int warp = tid / warp_size;
//
//    // Scan each warp using shfl_add.
//    type_t warp_scan = x;
//    iterate<s_log2(warp_size)>([&](int pass)
//    {
//      warp_scan = shfl_up_op(warp_scan, 1<< pass, op, warp_size);
//    });
//
//    // Store the intra-warp scans.
//    storage.threads[tid] = warp_scan;
//
//    // Store the reduction (last element) of each warp into storage.
//    if(min(warp_size * (warp + 1), count) - 1 == tid)
//      storage.warps[warp] = warp_scan;
//    __syncthreads();
//
//    // Scan the warp reductions.
//    if(tid < num_warps) { 
//      type_t cta_scan = storage.warps[tid];
//      iterate<s_log2(num_warps)>([&](int pass) {
//        cta_scan = shfl_up_op(cta_scan, 1<< pass, op, num_warps);
//      });
//      storage.warps[tid] = cta_scan;
//    }
//    __syncthreads();
//
//    type_t scan = warp_scan;
//    if(mgpu::scan_type_exc == type) {
//      scan = tid ? storage.threads[tid - 1] : init;
//      warp = (tid - 1) / warp_size;
//    }
//    if(warp > 0) scan = op(scan, storage.warps[warp - 1]);
//
//    type_t reduction = storage.warps[div_up(count, warp_size) - 1];
//    
//    my_scan_result_t<type_t> result { 
//      tid < count ? scan : reduction, 
//      reduction 
//    };
//    __syncthreads();
//
//    return result;
//  }
//
//#else

  //////////////////////////////////////////////////////////////////////////////
  // Standard CTA scan code that does not use shfl intrinsics. 

  template<class BinaryOperation>
  __device__
  my_scan_result_t<type_t>
    exclusive_scan(int tid, type_t x, storage_t& storage, int count, type_t init, BinaryOperation binary_op) const
  {
    // the first agent accumulates init into its summand
    if(tid == 0)
    {
      x = binary_op(init, x);
    }

    // all agents store their summand to temporary storage
    storage.data[tid] = x;

    __syncthreads();

    // double buffer to eliminate one barrier in the loop below 
    int first = 0;

    for(int offset = 1; offset < nt; offset += offset)
    {
      if(tid >= offset)
      {
        x = binary_op(storage.data[first + tid - offset], x);
      }

      first = nt - first;

      storage.data[first + tid] = x;

      __syncthreads();
    }

    my_scan_result_t<type_t> result;
    result.reduction = storage.data[first + count - 1];

    if(tid < count)
    {
      if(tid == 0)
      {
        result.scan = init;
      }
      else
      {
        result.scan = storage.data[first + tid - 1];
      }
    }
    else
    {
      result.scan = result.reduction;
    }

    __syncthreads();

    return result;
  }

//#endif  

  //////////////////////////////////////////////////////////////////////////////
  // CTA vectorized scan. Accepts multiple values per thread and adds in 
  // optional global carry-in.

  template<size_t vt, typename op_t = plus_t<type_t>>
  __device__
  my_scan_result_t<type_t, vt>
    inclusive_scan(int tid, agency::experimental::array<type_t, vt> local_array, storage_t& storage, type_t carry_in, int count = nt, op_t op = op_t()) const
  {
    // XXX this simpler implementation computes the same result as what is used below, but it requires additional state
    //     e.g., local_sum. the additional state requires more registers
    //// each agent accumulates the sum of at most vt elements in local_array
    //auto local_sum = ::accumulate_nonempty(bound<vt>(), local_array, op);

    //// exclusive scan the local sums
    //my_scan_result_t<type_t> result = exclusive_scan(tid, local_sum, storage, div_up(count, vt), carry_in, op);

    //// each agent computes an inclusive scan of its local array
    //::inclusive_scan(bound<vt>(), local_array, local_array, op, result.scan);

    //return my_scan_result_t<type_t, vt>{ local_array, result.reduction };

    // each agent does an in-place inclusive sum of at most vt elements in local_array
    ::inclusive_scan(bound<vt>(), local_array, local_array, op);

    // exclusive scan the thread-local sums to produce a carry-in for each thread
    my_scan_result_t<type_t> result = exclusive_scan(tid, local_array[vt - 1], storage, div_up<int>(count, vt), carry_in, op);

    // to produce the final ixclusive scan, add in the thread's carry-in
    for(int i = 0; i < vt; ++i)
    {
      local_array[i] = op(result.scan, local_array[i]);
    }

    return my_scan_result_t<type_t, vt>{ local_array, result.reduction };
  }

  template<size_t vt, typename op_t = plus_t<type_t>>
  __device__
  my_scan_result_t<type_t, vt>
    exclusive_scan(int tid, agency::experimental::array<type_t, vt> local_array, storage_t& storage, type_t carry_in = type_t(), int count = nt, op_t op = op_t()) const
  {
    // XXX this simpler implementation computes the same result as what is used below, but it requires additional state
    //     e.g., local_sum. the additional state requires more registers
    // each agent accumulates the sum of at most vt elements in local_array
    //auto local_sum = ::accumulate_nonempty(bound<vt>(), local_array, op);

    //// exclusive scan the local sums
    //my_scan_result_t<type_t> result = exclusive_scan(tid, local_sum, storage, div_up(count, vt), carry_in, op);

    //// each agent computes an exclusive scan of its local array
    //::exclusive_scan(bound<vt>(), local_array, local_array, result.scan, op);

    //return my_scan_result_t<type_t, vt>{ local_array, result.reduction };

    // each agent does an in-place inclusive sum of at most vt elements in local_array
    ::inclusive_scan(bound<vt>(), local_array, local_array, op);

    // exclusive scan the thread-local sums to produce a carry-in for each thread
    my_scan_result_t<type_t> result = exclusive_scan(tid, local_array[vt - 1], storage, div_up<int>(count, vt), carry_in, op);

    // to produce the final exclusive scan, shift the local_array right one slot and add in the thread's carry-in
    auto prev_plus_carry = result.scan;
    for(int i = 0; i < vt; ++i)
    {
      auto tmp = local_array[i];
      local_array[i] = prev_plus_carry;
      prev_plus_carry = op(result.scan, tmp);
    }

    return my_scan_result_t<type_t, vt>{ local_array, result.reduction };
  }
};

//////////////////////////////////////////////////////////////////////////////////
//// Overload for scan of bools.
//
//template<int nt>
//struct my_cta_scan_t<nt, bool>
//{
//  enum { num_warps = nt / warp_size };
//  struct storage_t {
//    int warps[num_warps];
//  };
//
//  MGPU_DEVICE my_scan_result_t<int> scan(int tid, bool x, 
//    storage_t& storage) const {
//
//    // Store the bit totals for each warp.
//    int lane = (warp_size - 1) & tid;
//    int warp = tid / warp_size;
//
//    int bits = __ballot(x);
//    storage.warps[warp] = popc(bits);
//    __syncthreads();
//
//#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
//    if(tid < num_warps) {
//      // Cooperative warp scan of partial reductions.
//      int scan = storage.warps[tid];
//      iterate<s_log2(num_warps)>([&](int i) {
//        scan = shfl_up_op(scan, 1<< i, plus_t<int>(), num_warps);
//      });
//      storage.warps[tid] = scan;
//    }
//    __syncthreads();
//#else
//    
//    if(0 == tid) {
//      // Inclusive scan of partial reductions..
//      int scan = 0;
//      iterate<num_warps>([&](int i) {
//        storage.warps[i] = scan += storage.warps[i];
//      });
//    }
//    __syncthreads();
//
//#endif    
//
//    int scan = ((warp > 0) ? storage.warps[warp - 1] : 0) +
//      popc(bfe(bits, 0, lane));
//    int reduction = storage.warps[num_warps - 1];
//    __syncthreads();
//
//    return my_scan_result_t<int> { scan, reduction };
//  }
//};

