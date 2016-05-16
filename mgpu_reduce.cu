#include <moderngpu/transform.hxx>   // for cta_launch.
#include <moderngpu/memory.hxx>      // for mem_t.
#include <moderngpu/cta_reduce.hxx>
#include "measure_bandwidth_of_invocation.hpp"
#include <cstdio>
#include <iostream>

using namespace mgpu;

template<typename launch_arg_t = empty_t, typename input_it, 
  typename output_it, typename op_t>
void reduce(input_it input, int count, output_it reduction, op_t op, 
  context_t& context) {

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_params_t<128, 8>
  >::type_t launch_t;

  typedef typename std::iterator_traits<input_it>::value_type type_t;

  int num_ctas = launch_t::cta_dim(context).num_ctas(count);
  mem_t<type_t> partials(num_ctas, context);
  type_t* partials_data = partials.data();

  auto k = [=] MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };
    typedef cta_reduce_t<nt, type_t> reduce_t;
    __shared__ typename reduce_t::storage_t shared_reduce;

    // Load the data for the first tile for each cta.
    range_t tile = get_tile(cta, nv, count);
    array_t<type_t, vt> x = mem_to_reg_strided<nt, vt>(input + tile.begin, 
      tid, tile.count());

    // Reduce the multiple values per thread into a scalar.
    type_t scalar;
    strided_iterate<nt, vt>([&](int i, int j) {
      scalar = i ? op(scalar, x[i]) : x[0];
    }, tid, tile.count());

    // Reduce to a scalar per CTA.
    scalar = reduce_t().reduce(tid, scalar, shared_reduce, 
      min(tile.count(), (int)nt), op, false);

    if(!tid) {
      if(1 == num_ctas) *reduction = scalar;
      else partials_data[cta] = scalar;
    }
  };
  cta_launch<launch_t>(k, num_ctas, context);

  // Recursively call reduce until there's just one scalar.
  if(num_ctas > 1)
    reduce<launch_params_t<512, 4> >(partials_data, num_ctas, reduction, op, 
      context);
}

int main(int argc, char** argv)
{
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
  reduce(input_device.data(), input_device.size(), output_device.data(), plus_t<int>(), context);

  // Get the reduction.
  std::vector<int> output_host = from_mem(output_device);

  // compare to reference
  assert(std::accumulate(input_host.begin(), input_host.end(), 0, std::plus<int>()) == output_host[0]);

  auto bandwidth = measure_bandwidth_of_invocation_in_gigabytes_per_second(100, sizeof(int) * n, [&]
  {
    reduce(input_device.data(), input_device.size(), output_device.data(), plus_t<int>(), context);
  });

  std::cout << "Mean bandwidth: " << bandwidth << " GB/s " << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

