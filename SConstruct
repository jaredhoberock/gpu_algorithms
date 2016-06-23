env = Environment(tools = ['default', 'nvcc'])

env.MergeFlags(['-O3', '-std=c++11', '-lstdc++', '-lpthread', '-Isubmodules/moderngpu/src', '-Isubmodules/agency', '-Isubmodules/time_invocation'])

# next, flags for nvcc
env.MergeFlags(['--expt-extended-lambda', '-arch=sm_52', '-Xptxas=--verbose'])

# reduction implementations
env.Program('mgpu_reduce.cu')
env.Program('agency_mgpu_reduce.cu')
env.Program('agency_reduce.cu')

# scan implementations
env.Program('mgpu_scan.cu')
env.Program('agency_mgpu_scan.cu')
env.Program('agency_scan.cu')

env.Program('mgpu_downsweep.cu')
env.Program('agency_mgpu_downsweep.cu')
env.Program('agency_downsweep.cu')

