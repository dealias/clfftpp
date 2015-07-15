#include <stdlib.h>
#include <platform.hpp>
#include <clfft.hpp>
#include <iostream>
#include "utils.hpp"

int main() {
  int platnum = 0;
  int devnum = 0;
  unsigned int nx = 4;
  unsigned int ny = 4;
  bool inplace = false;

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
  cl_mem inbuf, outbuf;

  if(inplace) {
    fft.create_cbuf(&inbuf);
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    fft.create_rbuf(&inbuf);
    fft.create_cbuf(&outbuf);
  }

  std::string init_source ="\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n	\
__kernel void init(__global double *X, const unsigned int ny)		\
{						\
  const int i = get_global_id(0);		\
  const int j = get_global_id(1);		\
  unsigned pos = i * ny + j;			\
  X[pos] = i * i + j;				\
}";
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  set_kernel_arg(initkernel, 0, sizeof(cl_mem), &inbuf);
  set_kernel_arg(initkernel, 1, sizeof(unsigned int), &ny);

  std::cout << "Allocating " 
	    << (inplace ? 2 * fft.ncomplex() : fft.nreal())
	    << " doubles for real." << std::endl;
  double *Xin = new double[inplace ? 2 * fft.ncomplex() : fft.nreal()];
  std::cout << "Allocating "
	    << 2 * fft.ncomplex() 
	    << " doubles for complex." << std::endl;
  double *Xout = new double[2 * fft.ncomplex()];

  // Create OpenCL events
  cl_event clv_init = clCreateUserEvent(ctx, NULL);
  cl_event clv_toram = clCreateUserEvent(ctx, NULL);
  cl_event clv_forward = clCreateUserEvent(ctx, NULL);
  cl_event clv_backward = clCreateUserEvent(ctx, NULL);

  std::cout << "\nInput:" << std::endl;
  size_t global_wsize[] = {nx, ny};
  clEnqueueNDRangeKernel(queue,
			 initkernel,
			 2, // cl_uint work_dim,
			 NULL, // global_work_offset,
			 global_wsize, // global_work_size, 
			 NULL, // size_t *local_work_size, 
			 0, NULL, &clv_init);
  fft.rbuf_to_ram(Xin, &inbuf, 1, &clv_init, &clv_toram);
  show2R(Xin, nx, ny);

  std::cout << "\nTransformed:" << std::endl;
  fft.forward(&inbuf, inplace ? NULL : &outbuf, 1, &clv_init, &clv_forward);
  fft.cbuf_to_ram(Xout, inplace ? &inbuf : &outbuf, 
		  1, &clv_forward, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show2H(Xout, fft.ncomplex(0), fft.ncomplex(1), inplace ? 1 : 0);

  std::cout << "\nTransformed back:" << std::endl;
  fft.backward(inplace ? &inbuf : &outbuf, 
	       inplace ? NULL : &inbuf, 1, &clv_forward, &clv_backward);
  fft.rbuf_to_ram(Xin, &inbuf, 1, &clv_backward, &clv_toram);
  clWaitForEvents(1, &clv_toram);
  show2R(Xin, nx, ny);

  delete[] Xout;
  delete[] Xin;
  
  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
