#include <stdlib.h>
#include <iostream>
#include <platform.hpp>
#include <clfft.hpp>
#include "utils.hpp"

int main() {
  int platnum = 1;
  int devnum = 0;

  bool inplace = true;

  unsigned int nx = 4;
  unsigned int ny = 4;
  unsigned int nz = 4;

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
  
  clfft3 fft(nx, ny, nz, inplace, queue, ctx);

  unsigned int ncomplex = nx * ny * nz;

  cl_int status;
  cl_mem inbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				sizeof(double) * 2 * ncomplex, NULL,
				&status);
  cl_mem outbuf;
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    outbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				   sizeof(double) * 2 * ncomplex, NULL,
			    &status);
  }

  std::string init_source = "\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n	\
__kernel void init(__global double *X,		\
const unsigned int ny, const unsigned int nz)\n	\
{\n						\
  const int i = get_global_id(0);\n		\
  const int j = get_global_id(1);\n		\
  const int k = get_global_id(2);\n		\
  const int pos = i * nz * ny + j * nz + k;	\
  X[2 * pos] = i;				\
  X[2 * pos + 1] = j + k * k;			\
}\n";
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  set_kernel_arg(initkernel, 0, sizeof(cl_mem), &inbuf);
  set_kernel_arg(initkernel, 1, sizeof(unsigned int), &ny);
  set_kernel_arg(initkernel, 2, sizeof(unsigned int), &nz);

  std::cout << "Allocating " << 2 * ncomplex << " doubles." << std::endl;
  double *X = new double[2 * ncomplex];
  double *Xout = new double[2 * ncomplex];

  cl_event clv_init;
  cl_event clv_toram;
  cl_event clv_forward;
  cl_event clv_backward;

  std::cout << "\nInput:" << std::endl;
  size_t global_wsize[] = {nx, ny, nz};
  clEnqueueNDRangeKernel(queue, initkernel, 3, NULL, global_wsize, NULL,
			 0, NULL, &clv_init);
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, sizeof(double) * 2 *ncomplex, X,
		      1, &clv_init, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show3C(X, nx, ny, nz);

  std::cout << "\nTransformed:" << std::endl;
  fft.forward(&inbuf, inplace ? NULL : &outbuf, 1, &clv_init, &clv_forward);    
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, sizeof(double) * 2 *ncomplex, X,
		      1, &clv_forward, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show3C(Xout, nx, ny, nz);

  std::cout << "\nTransformed back:" << std::endl;
  fft.backward(inplace ? &inbuf : &outbuf, 
	       inplace ? NULL : &inbuf, 1, &clv_forward, &clv_backward);
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, sizeof(double) * 2 *ncomplex, X,
		      1, &clv_backward, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show3C(X, nx, ny, nz);
  
  delete X;

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
