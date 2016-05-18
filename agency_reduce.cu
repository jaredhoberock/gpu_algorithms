#include <iostream>
#include <moderngpu/transform.hxx>   // for cta_launch.
#include <moderngpu/memory.hxx>      // for mem_t.
#include <moderngpu/cta_reduce.hxx>
#include <agency/agency.hpp>
#include <agency/experimental/strided_view.hpp>
#include <agency/experimental/span.hpp>
#include <agency/experimental/array.hpp>
#include <agency/cuda.hpp>
#include "measure_bandwidth_of_invocation.hpp"
#include "bound.hpp"
#include "algorithm.hpp"
#include <cstdio>

auto grid(int num_blocks, int num_threads) ->
  decltype(agency::cuda::par(num_blocks, agency::cuda::con(num_threads)))
{
  return agency::cuda::par(num_blocks, agency::cuda::con(num_threads));
}

using grid_agent = agency::parallel_group<agency::cuda::concurrent_agent>;


template<size_t i, size_t count, bool valid = (i < count)>
struct my_iterate_t
{
  #pragma nv_exec_check_disable
  template<typename func_t>
  __host__ __device__ static void eval(func_t f)
  {
    f(i);
    my_iterate_t<i + 1, count>::eval(f);
  }
};

template<size_t i, size_t count>
struct my_iterate_t<i, count, false>
{
  template<typename func_t>
  __host__ __device__ static void eval(func_t f) { }
};

template<size_t begin, size_t end, typename func_t>
__host__ __device__ void my_iterate(func_t f)
{
  my_iterate_t<begin, end>::eval(f);
}

template<size_t count, typename func_t>
__host__ __device__ void my_iterate(func_t f)
{
//  unrolling_executor<count> exec;
//
//  exec.execute(f, count);

  my_iterate_t<0, count>::eval(f);
}


// Invoke unconditionally.
template<int nt, int vt, typename func_t>
__device__ void my_strided_iterate(func_t f, int tid)
{
  my_iterate<vt>([=](int i)
  {
    f(i, nt * i + tid);
  });
}

// Check range.
template<int nt, int vt, int vt0 = vt, typename func_t>
__device__ void my_strided_iterate(func_t f, int tid, int count)
{
  // Unroll the first vt0 elements of each thread.
  if(vt0 > 1 && count >= nt * vt0)
  {
    my_strided_iterate<nt, vt0>(f, tid);    // No checking
  }
  else
  {
    my_iterate<vt0>([=](int i)
    {
      int j = nt * i + tid;
      if(j < count) f(i, j);
    });
  }

  my_iterate<vt0, vt>([=](int i)
  {
    int j = nt * i + tid;
    if(j < count) f(i, j);
  });
}


template<int nt, int vt, int vt0 = vt, typename it_t>
__host__ __device__ mgpu::array_t<typename std::iterator_traits<it_t>::value_type, vt> 
my_mem_to_reg_strided(it_t mem, int tid, int count)
{
  typedef typename std::iterator_traits<it_t>::value_type type_t;
  mgpu::array_t<type_t, vt> x;
  my_strided_iterate<nt, vt, vt0>([&](int i, int j)
  { 
    x[i] = mem[j]; 
  }, tid, count);

  return x;
}


template<typename launch_arg_t = mgpu::empty_t, typename input_it,  typename output_it, typename op_t>
void my_reduce(input_it input, int count, output_it reduction, op_t op, mgpu::context_t& context)
{
  using namespace mgpu;
  using namespace agency::experimental;

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_params_t<128, 8>
  >::type_t launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);
  int num_threads = launch_t::cta_dim(context).nt;
  mem_t<type_t> partials(num_ctas, context);
  type_t* partials_data = partials.data();

  auto k = [=] __device__ (int tid, int cta)
  {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
    typedef cta_reduce_t<nt, type_t> reduce_t;
    __shared__ typename reduce_t::storage_t shared_reduce;

    // Load the data for the first tile for each cta.
    range_t tile = get_tile(cta, nv, count);

    // stride through the input and compute a partial sum per thread
    span<type_t> our_span(input + tile.begin, input + tile.end);
    auto my_values = strided(drop(our_span, tid), size_t(nt));

    // XXX use optional here?
    type_t partial_sum;
    if(!my_values.empty())
    {
      type_t init = my_values[0];
      partial_sum = reduce(bound<vt-1>(), drop(my_values, 1), init, op);
    }

    // Reduce to a scalar per CTA.
    // XXX what if my_values was empty? what do I use for my partial?
    int num_partials = min(tile.count(), (int)nt);
    auto result = reduce_t().reduce(tid, partial_sum, shared_reduce, num_partials, op, false);

    if(tid == 0)
    {
      if(1 == num_ctas) *reduction = result;
      else partials_data[cta] = result;
    }
  };

  agency::bulk_invoke(grid(num_ctas, num_threads), [=] __device__ (grid_agent& self)
  {
    k(self.inner().index(), self.outer().index());
  });

  // Recursively call reduce until there's just one scalar.
  if(num_ctas > 1)
  {
    my_reduce<launch_params_t<512, 4> >(partials_data, num_ctas, reduction, op, context);
  }
}

int main(int argc, char** argv)
{
  using namespace mgpu;

  standard_context_t context;

  size_t n = 1 << 30;

  // Prepare the fibonacci numbers on the host.
  std::vector<int> input_host(n);
  for(int i = 0; i < input_host.size(); ++i)
    input_host[i] = (i + 1) * (i + 1);

  // Copy the data to the GPU.
  mem_t<int> input_device = to_mem(input_host, context);

  // Call our simple reduce.
  mem_t<int> output_device(1, context);
  my_reduce(input_device.data(), input_device.size(), output_device.data(), plus_t<int>(), context);

  // Get the reduction.
  std::vector<int> output_host = from_mem(output_device);

  // compare to reference
  assert(std::accumulate(input_host.begin(), input_host.end(), 0, std::plus<int>()) == output_host[0]);

  auto bandwidth = measure_bandwidth_of_invocation_in_gigabytes_per_second(100, sizeof(int) * n, [&]
  {
    my_reduce(input_device.data(), input_device.size(), output_device.data(), plus_t<int>(), context);
  });

  std::cout << "Mean bandwidth: " << bandwidth << " GB/s " << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

