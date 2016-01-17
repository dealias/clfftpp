#include <stdlib.h>
#include <platform.hpp>
#include <clfft.hpp>
#include <iostream>
#include "utils.hpp"

int main() {
  int platnum = 1;
  int devnum = 0;

  unsigned int nx = 4;

  bool inplace = false;

  unsigned int nxp = nx / 2 + 1;
  unsigned int nreal = inplace ? 2 * nxp : nx;
  
  platform::show_devices();
  std::cout << "Using platform " << platnum
	    << " device " << devnum 
	    << "." << std::endl;

  std::vector<std::vector<cl_device_id> > dev_ids;
  platform::create_device_tree(dev_ids);
  cl_device_id device = dev_ids[platnum][devnum];

  std::vector<cl_platform_id> plat_ids;
  platform::find_platform_ids(plat_ids);
  cl_platform_id platform = plat_ids[platnum];

  cl_context ctx = platform::create_context(platform, device);
  cl_command_queue queue = platform::create_queue(ctx, device,
						  CL_QUEUE_PROFILING_ENABLE);

  clfft1r fft(nx, inplace, queue, ctx);

  cl_int status;
  cl_mem inbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				sizeof(double) * nreal, NULL, &status);
  cl_mem outbuf;
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    outbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				   sizeof(double) * 2 * nxp, NULL, &status);
  }

  // Create OpenCL kernel to initialize OpenCL buffer
  std::string init_source = "\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n	\
__kernel void init(__global double *X)\n	\
{\n						\
  const int i = get_global_id(0);\n		\
  X[i] = i;\n					\
}\n";
  cl_program initprog = platform::create_program(init_source, ctx);
  clBuildProgram(initprog, 1, &device, NULL, NULL, NULL);
  cl_kernel initkernel = clCreateKernel(initprog, "init", &status); 
  clSetKernelArg(initkernel, 0, sizeof(cl_mem), &inbuf);

  std::cout << "Allocating " << nreal << " doubles for real." << std::endl;
  double *X = new double[nreal];
  std::cout << "Allocating " << 2 * nxp << " doubles for complex." << std::endl;
  double *FX = new double[2 * nxp];

  cl_event clv_init;
  cl_event clv_toram;
  cl_event clv_forward;
  cl_event clv_backward;

  std::cout << "\nInput:" << std::endl;
  size_t global_wsize[] = {(size_t)nx};
  clEnqueueNDRangeKernel(queue, initkernel, 1, NULL, global_wsize, NULL,
			 0, NULL, &clv_init);
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, sizeof(double) * nreal, X,
		      1, &clv_init, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show1R(X, nx);

  std::cout << "\nTransformed:" << std::endl;
  fft.forward(&inbuf, inplace ? NULL : &outbuf, 1, &clv_init, &clv_forward);
  clEnqueueReadBuffer(queue, inplace ? inbuf : outbuf,
		      CL_TRUE, 0, sizeof(double) * 2 * nxp, FX,
		      1, &clv_init, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show1C(FX, nxp);

  std::cout << "\nTransformed back:" << std::endl;
  fft.backward(inplace ? &inbuf : &outbuf, 
	       inplace ? NULL : &inbuf, 1, &clv_forward, &clv_backward);
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, sizeof(double) * nreal, X,
		      1,  &clv_backward, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show1R(X, nx);
  
  delete[] X;
  delete[] FX;

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
