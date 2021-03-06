/*
This file is part of clFFT++.

clFFT++ is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

clFFT++ is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with clFFT++.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdlib.h>
#include <iostream>
#include <platform.hpp>
#include <clfft++.hpp>
#include "utils.hpp"

int main() {
  int platnum = 0;
  int devnum = 0;

  bool inplace = false;

  // Input buffer size
  unsigned int nx = 4;
  unsigned int ny = 4;
  std::cout << "nx: " << nx << "\tny: " << ny << std::endl;

  // Length of transform:
  unsigned int n = 4;
  // Number of transforms:
  unsigned int M = 4; 
  std::cout << "n: " << n << "\tM: " << M << std::endl;

#if 1
  // transform in second index
  size_t istride = 1;
  size_t ostride = 1;
  size_t idist = nx;
  size_t odist = nx;
#else
  // transform in first index
  size_t istride = nx;
  size_t ostride = nx;
  size_t idist = 1;
  size_t odist = 1;
#endif
  
  // Set up OpenCL environment
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

  
  // Create fft
  clfftpp::clmfft1 fft(n, M, istride, ostride, idist, odist, inplace,
		       queue, ctx);

  // Set up buffers
  std::cout << "Allocating " << 2 * nx * ny << " doubles." << std::endl;
  double *X = new double[2 * nx * ny];
  double *FX = new double[2 * n * M];
  size_t bufsize = sizeof(double) * 2 * nx * ny;
  cl_int status;
  cl_mem inbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bufsize, NULL, &status);
  cl_mem outbuf;
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    outbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bufsize, NULL, &status);
  }
  
  // Set up initialization kernel
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
  cl_program initprog = platform::create_program(init_source, ctx);
  clBuildProgram(initprog, 1, &device, NULL, NULL, NULL);
  cl_kernel initkernel = clCreateKernel(initprog, "init", &status); 
  clSetKernelArg(initkernel, 0, sizeof(cl_mem), &inbuf);
  size_t global_wsize[] = {nx, ny};
  

  std::cout << "\nInput:" << std::endl;
  clEnqueueNDRangeKernel(queue, initkernel, 2, NULL, global_wsize, NULL, 
			 0, 0, 0);
  clFinish(queue);
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, bufsize, X, 0, 0, 0);
  clFinish(queue);
  show2C(X, nx, ny);

  std::cout << "\nTransformed:" << std::endl;
  fft.forward(&inbuf, inplace ? NULL : &outbuf, 0, 0, 0); 
  clFinish(queue);   
  clEnqueueReadBuffer(queue, inplace ? inbuf : outbuf, CL_TRUE, 0, bufsize, FX,
		      0, 0, 0);
  clFinish(queue);
  show2C(FX, n, M);

  
  std::cout << "\nTransformed back:" << std::endl;
  fft.backward(inplace ? &inbuf : &outbuf, inplace ? NULL : &inbuf, 0, 0, 0);
  clFinish(queue);
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, bufsize, X,  0, 0, 0);
  clFinish(queue);
  show2C(X, nx, M);
  

  // Clean up
  delete X;
  delete FX;
  clReleaseMemObject(inbuf);
  if(!inplace)
    clReleaseMemObject(outbuf);
  
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
