#include <stdlib.h>
#include <platform.hpp>
#include <clfft.hpp>
#include <iostream>
#include "utils.hpp"

int main() {
  // OpenCL platform / device choice
  int platnum = 1;
  int devnum = 0;

  // Input buffer size
  unsigned int nx = 4;
  unsigned int ny = 4;

  unsigned int n = 4; // lenght of transform
  unsigned int M = 4; // number of transforms in batch
  
  bool inplace = false;

  unsigned int np = n / 2 + 1;
  unsigned int ncomplex = np * M;

#if 1
  // transform along index 0
  size_t istride = nx;
  size_t ostride = nx;
  size_t idist = 1;
  size_t odist = 1;
  unsigned int nreal = inplace ? 2 * (nx / 2 + 1) * ny : nx * ny;
  size_t global_wsize[2] = {inplace ? 2 * (nx / 2 + 1) : nx, ny};
  unsigned int cdim[2] = {np, M};
#else
  // transform along index 1
  size_t istride = 1;
  size_t idist = nx;
  size_t ostride = 1;
  size_t odist = nx / 2 + 1;
  unsigned int nreal = inplace ? nx * 2 * (ny / 2 + 1) : nx * ny;
  size_t global_wsize[2] = {nx, inplace ?2 * (ny / 2 + 1) : ny};
  unsigned int cdim[2] = {M, np};
#endif
  
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

  clmfft1r fft(n, M, istride, ostride, idist, odist, inplace, queue, ctx);

  cl_int status;
  cl_mem inbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				sizeof(double) * 2 * nx * ny, NULL, &status);
  cl_mem outbuf;
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    outbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			    sizeof(double) * 2 * ncomplex, NULL, &status);
  }

  std::string init_source ="\
#pragma OPENCL EXTENSION cl_khr_fp64: enable \n			\
__kernel void init(__global double *X,		\n		\
      const unsigned int nx, const unsigned int ny ) \n		\
{						\n		\
  const int i = get_global_id(0); 		\n		\
  const int j = get_global_id(1); 		\n		\
  const int stride = get_global_size(1); 	\n		\
  int pos = i * stride + j;			\n		\
  if((i < nx) && (j < ny))			\n		\
    X[pos] = j + 10.0 * i;			\n		\
  else						\n		\
    X[pos] = 0.0;				\n		\
}";
   
  cl_program initprog = create_program(init_source, ctx);
  clBuildProgram(initprog, 1, &device, NULL, NULL, NULL);
  cl_kernel initkernel = clCreateKernel(initprog, "init", &status); 
  clSetKernelArg(initkernel, 0, sizeof(cl_mem), &inbuf);
  clSetKernelArg(initkernel, 1, sizeof(unsigned int), &nx);
  clSetKernelArg(initkernel, 2, sizeof(unsigned int), &ny);

  std::cout << "Allocating " << nreal << " doubles for real." << std::endl;
  double *X = new double[nreal];
  std::cout << "Allocating " << 2 * ncomplex
	    << " doubles for complex." << std::endl;
  double *FX = new double[2 * ncomplex];

  // Create OpenCL events
  cl_event clv_init;
  cl_event clv_toram;
  cl_event clv_forward;
  cl_event clv_backward;

  std::cout << "\nInput:" << std::endl;
  clEnqueueNDRangeKernel(queue, initkernel, 2, NULL, global_wsize, NULL,
			 0, NULL, &clv_init);
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, sizeof(double) * nreal, X,
		 1, &clv_init, &clv_toram);
  show1R(X, nx, ny);

  std::cout << "\nTransformed:" << std::endl;
  fft.forward(&inbuf, inplace ? NULL : &outbuf, 1, &clv_init, &clv_forward);
  clEnqueueReadBuffer(queue, inplace ? inbuf : outbuf,
		      CL_TRUE, 0, sizeof(double) * 2 * ncomplex,
		      FX,
		      1, &clv_forward, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show2C(FX, cdim[0], cdim[1]);

  std::cout << "\nTransformed back:" << std::endl;
  fft.backward(inplace ? &inbuf : &outbuf, 
  	       inplace ? NULL : &inbuf, 1, &clv_forward, &clv_backward);
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, sizeof(double) * nreal, X, 1,
		      &clv_backward, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show1R(X, n, M);

  delete[] FX;
  delete[] X;
  
  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
