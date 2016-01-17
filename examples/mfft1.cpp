#include <stdlib.h>
#include <iostream>
#include <platform.hpp>
#include <clfft.hpp>
#include "utils.hpp"

int main() {
  int platnum = 1;
  int devnum = 0;
  bool inplace = true;

  // Input buffer size
  unsigned int nx = 4;
  unsigned int ny = 4;
  std::cout << "nx: " << nx << "\tny: " << ny << std::endl;
  
  unsigned int n = 4; // length of transform
  unsigned int M = 4; // number of transforms in batch
  std::cout << "n: " << n << "\tM: " << M << std::endl;
  
  // transform in second index
  // size_t istride = 1;
  // size_t ostride = 1;
  // size_t idist = nx;
  // size_t odist = nx;

  // transform in first index
  size_t istride = nx;
  size_t ostride = nx;
  size_t idist = 1;
  size_t odist = 1;

  show_devices();
  std::cout << "Using platform " << platnum
	    << " device " << devnum 
	    << "." << std::endl;
  
  std::vector<std::vector<cl_device_id> > dev_ids;
  create_device_tree(dev_ids);
  cl_device_id device = dev_ids[platnum][devnum];

  std::vector<cl_platform_id> plat_ids;
  find_platform_ids(plat_ids);
  cl_platform_id platform = plat_ids[platnum];

  cl_context ctx = create_context(platform, device);
  cl_command_queue queue = create_queue(ctx, device, CL_QUEUE_PROFILING_ENABLE);
  
  clmfft1 fft(n, M, istride, ostride, idist, odist, inplace, queue, ctx);

  cl_int status;
  cl_mem inbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				sizeof(double) * 2 * nx * ny, NULL,
				&status);
  cl_mem outbuf;
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    outbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			    2 * sizeof(double) * nx * ny, NULL, &status);
  }
  
  std::string init_source = "\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n	\
__kernel void init(__global double *X)		\
{						\
  const int i = get_global_id(0);		\
  const int j = get_global_id(1);		\
  const int ny = get_global_size(1);		\
  int pos = i * ny + j;				\
  X[2 * pos] = i;				\
  X[2 * pos + 1] = j;				\
}";
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  set_kernel_arg(initkernel, 0, sizeof(cl_mem), &inbuf);
  size_t global_wsize[] = {nx, ny};
  
  std::cout << "Allocating " << 2 * nx * ny << " doubles." << std::endl;
  double *X = new double[2 * nx * ny];
  double *FX = new double[2 * n * M];

  cl_event clv_init;
  cl_event clv_toram;
  cl_event clv_forward;
  cl_event clv_backward;

  std::cout << "\nInput:" << std::endl;
  clEnqueueNDRangeKernel(queue, initkernel, 2, NULL, global_wsize, NULL, 
			 0, NULL, &clv_init);
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, sizeof(double) * 2 * nx * ny, X,
		      1, &clv_init, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show2C(X, nx, ny);

  std::cout << "\nTransformed:" << std::endl;
  fft.forward(&inbuf, inplace ? NULL : &outbuf, 1, &clv_init, &clv_forward);    
  clEnqueueReadBuffer(queue, inplace ? inbuf : outbuf,
		      CL_TRUE, 0, sizeof(double) * 2 * nx * ny, FX,
		      1, &clv_forward, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show2C(FX, n, M);

  std::cout << "\nTransformed back:" << std::endl;
  fft.backward(inplace ? &inbuf : &outbuf, 
	       inplace ? NULL : &inbuf, 1, &clv_forward, &clv_backward);
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, sizeof(double) * 2 * nx * ny, X,
		      1, &clv_backward, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show2C(X, nx, M);
  
  delete X;
  delete FX;

  clReleaseMemObject(inbuf);
  if(!inplace)
    clReleaseMemObject(outbuf);
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
