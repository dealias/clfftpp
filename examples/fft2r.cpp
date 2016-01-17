#include <stdlib.h>
#include <platform.hpp>
#include <clfft.hpp>
#include <iostream>
#include "utils.hpp"

int main() {
  int platnum = 1;
  int devnum = 0;

  unsigned int nx = 4;
  unsigned int ny = 4;

  bool inplace = true;

  unsigned int nyp = ny / 2 + 1;
  unsigned int ncomplex = nx * nyp;
  unsigned int nreal = inplace ? 2 * ncomplex : nx * ny;
  
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

  clfft2r fft(nx, ny, inplace, queue, ctx);

  cl_int status;
  cl_mem inbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				sizeof(double) * nreal, NULL, &status);
  cl_mem outbuf;
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    outbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			    sizeof(double) * 2 * ncomplex, NULL, &status);
  }

  int skip = inplace ? 2 * nyp : ny;
  std::string init_source ="\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n	\
__kernel void init(__global double *X, const unsigned int skip)		\
{						\
  const int i = get_global_id(0);		\
  const int j = get_global_id(1);		\
  unsigned pos = i * skip + j;			\
  X[pos] = i * i + j;				\
}";
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  clSetKernelArg(initkernel, 0, sizeof(cl_mem), &inbuf);
  clSetKernelArg(initkernel, 1, sizeof(unsigned int), &skip);

  std::cout << "Allocating " << nreal  << " doubles for real." << std::endl;
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
  size_t global_wsize[] = {nx, ny};
  clEnqueueNDRangeKernel(queue, initkernel, 2, NULL, global_wsize, NULL, 
			 0, NULL, &clv_init);
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, sizeof(double) * nreal, X,
		      1, &clv_init, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show2R(X, nx, ny, skip);

  std::cout << "\nTransformed:" << std::endl;
  fft.forward(&inbuf, inplace ? NULL : &outbuf, 1, &clv_init, &clv_forward);
  clEnqueueReadBuffer(queue, inplace ? inbuf : outbuf,
		      CL_TRUE, 0, sizeof(double) * 2 * ncomplex, FX,
		      1, &clv_forward, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show2C(FX, nx, nyp);

  std::cout << "\nTransformed back:" << std::endl;
  fft.backward(inplace ? &inbuf : &outbuf, 
	       inplace ? NULL : &inbuf, 1, &clv_toram, &clv_backward);
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, sizeof(double) * nreal, X,
		      1, &clv_backward, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show2R(X, nx, ny, skip);

  delete[] FX;
  delete[] X;

  clReleaseMemObject(inbuf);
  if(!inplace)
    clReleaseMemObject(outbuf);
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
