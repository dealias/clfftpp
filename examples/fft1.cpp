#include <stdlib.h>
#include <platform.hpp>
#include <clfft.hpp>

#include <iostream>

#include "utils.hpp"

template<class T>
void init(T *X, unsigned int n)
{
  for(unsigned int i = 0; i < n; ++i) {
    X[2 * i] = i;
    X[2 * i + 1] = 0.0;
  }
}

int main() {
  int platnum = 0;
  int devnum = 0;
  bool inplace = true;
  unsigned int nx = 4;

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
  
  clfft1 fft(nx, inplace, queue, ctx);

  cl_mem inbuf, outbuf;
  fft.create_cbuf(&inbuf);
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    fft.create_cbuf(&outbuf);
  }
  
  // Create OpenCL kernel to initialize OpenCL buffer
  std::string init_source = "\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n	\
__kernel void init(__global double *X)\n	\
{\n						\
  const int i = get_global_id(0);\n		\
  X[2 * i] = i;\n					\
  X[2 * i + 1] = 0;\n					\
}\n";
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  set_kernel_arg(initkernel, 0, sizeof(cl_mem), &inbuf);

  std::cout << "Allocating " 
	    << 2 * fft.ncomplex() 
	    << " doubles." << std::endl;
  double *X = new double[2 * fft.ncomplex()];

  cl_event clv_init = clCreateUserEvent(ctx, NULL);
  cl_event clv_toram = clCreateUserEvent(ctx, NULL);
  cl_event clv_forward = clCreateUserEvent(ctx, NULL);
  cl_event clv_backward = clCreateUserEvent(ctx, NULL);

  std::cout << "\nInput:" << std::endl;
  size_t global_wsize[] = {nx};
  clEnqueueNDRangeKernel(queue,
			 initkernel,
			 1, // cl_uint work_dim,
			 NULL, // global_work_offset,
			 global_wsize, // global_work_size, 
			 NULL, // size_t *local_work_size, 
			 0, NULL, &clv_init);
  fft.cbuf_to_ram(X, &inbuf, 1, &clv_init, &clv_toram);
  show1C(X, nx);

  fft.forward(&inbuf, inplace ? NULL : &outbuf, 1, &clv_init, &clv_forward);
  fft.cbuf_to_ram(X, inplace ? &inbuf : &outbuf, 1, &clv_forward, &clv_toram);
  clWaitForEvents(1, &clv_toram);

  std::cout << "\nTransformed:" << std::endl;
  show1C(X, nx);

  fft.backward(inplace ? &inbuf : &outbuf, 
	       inplace ? NULL : &inbuf, 1, &clv_forward, &clv_backward);    
  fft.cbuf_to_ram(X, &inbuf, 1, &clv_backward, &clv_toram);
  clWaitForEvents(1, &clv_toram);

  std::cout << "\nTransformed back:" << std::endl;
  show1C(X, nx);

  delete[] X;
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
