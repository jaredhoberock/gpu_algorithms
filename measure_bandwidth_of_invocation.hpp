#pragma once

#include <time_invocation.hpp>

template<class Function, class... Args>
double measure_bandwidth_of_invocation_in_gigabytes_per_second(std::size_t num_trials, std::size_t num_bytes, Function&& f, Args&&... args)
{
  auto nsecs = ::time_invocation_in_nanoseconds(num_trials, std::forward<Function>(f), std::forward<Args>(args)...);

  double seconds = double(nsecs) / 1000000000;

  double bytes_per_second = double(num_bytes) / seconds;

  double gigabytes_per_second = bytes_per_second / 1000000000;

  return gigabytes_per_second;
}

