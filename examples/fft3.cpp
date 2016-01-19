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

  bool inplace = true;

  unsigned int nx = 4;
  unsigned int ny = 4;
  unsigned int nz = 4;

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
  clfftpp::clfft3 fft(nx, ny, nz, inplace, queue, ctx);


  // Set up buffers
  std::cout << "Allocating " << 2 * nx * ny * nz << " doubles." << std::endl;
  double *X = new double[2 * nx * ny * nz];
  double *FX = new double[2 * nx * ny * nz];
  size_t bufsize = sizeof(double) * 2 * nx * ny * nz;
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
  const int k = get_global_id(2);		\
  const int ny = get_global_size(1);		\
  const int nz = get_global_size(2);		\
  const int pos = i * nz * ny + j * nz + k;	\
  X[2 * pos] = i;				\
  X[2 * pos + 1] = j + k * k;			\
}";
  cl_program initprog = platform::create_program(init_source, ctx);
  clBuildProgram(initprog, 1, &device, NULL, NULL, NULL);
  cl_kernel initkernel = clCreateKernel(initprog, "init", &status); 
  clSetKernelArg(initkernel, 0, sizeof(cl_mem), &inbuf);

  std::cout << "\nInput:" << std::endl;
  size_t global_wsize[] = {nx, ny, nz};
  clEnqueueNDRangeKernel(queue, initkernel, 3, NULL, global_wsize, NULL,
			 0, 0, 0);
  clFinish(queue);
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, bufsize, X, 0, 0, 0);
  clFinish(queue);
  show3C(X, nx, ny, nz);

  std::cout << "\nTransformed:" << std::endl;
  fft.forward(&inbuf, inplace ? NULL : &outbuf, 0, 0, 0);
  clFinish(queue);
  clEnqueueReadBuffer(queue, inplace ? inbuf : outbuf, CL_TRUE, 0, bufsize, FX,
		      0, 0, 0);
  clFinish(queue);
  show3C(FX, nx, ny, nz);

  std::cout << "\nTransformed back:" << std::endl;
  fft.backward(inplace ? &inbuf : &outbuf, inplace ? NULL : &inbuf, 0, 0, 0);
  clFinish(queue);
  clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, bufsize, X, 0, 0, 0);
  clFinish(queue);
  show3C(X, nx, ny, nz);

  // Clean up
  delete[] X;
  delete[] FX;

  clReleaseMemObject(inbuf);
  if(!inplace)
    clReleaseMemObject(outbuf);
  
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
