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
#include <platform.hpp>
#include <clfft++.hpp>

#include <iostream>
#include <timing.h>
#include <seconds.h>

#include <getopt.h>
#include "utils.hpp"

#include "Array.h"
#include "Complex.h"
#include "fftw++.h"

using namespace std;

int main(int argc, char *argv[]) {
  int platnum = 0;
  int devnum = 0;

  bool inplace = true;

  unsigned int nx = 4;
  unsigned int ny = 4;
  unsigned int nz = 4;

  unsigned int N = 0;

  unsigned int maxout = 10000;

  double tolerance = 1e-9;
  
  unsigned int stats = 0; // Type of statistics used in timing test.

  int error = 0;

#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"P:D:m:x:y:z:N:S:hi:");
    if (c == -1) break;
    
    switch (c) {
    case 'P':
      platnum = atoi(optarg);
      break;
    case 'D':
      devnum = atoi(optarg);
      break;
    case 'x':
      nx = atoi(optarg);
      break;
    case 'y':
      ny = atoi(optarg);
      break;
    case 'z':
      nz = atoi(optarg);
      break;
    case 'm':
      nx = atoi(optarg);
      ny = atoi(optarg);
      nz = atoi(optarg);
      break;
    case 'N':
      N = atoi(optarg);
      break;
    case 'S':
      stats = atoi(optarg);
      break;
    case 'i':
      inplace = atoi(optarg);
      break;
    case 'h':
      usage(3);
      exit(0);
      break;
    default:
      cout << "Invalid option" << endl;
      usage(3);
      exit(1);
    }
  }

  platform::show_devices();
  cout << "Using platform " << platnum
	    << " device " << devnum 
	    << "." << endl;
  
  vector<vector<cl_device_id> > dev_ids;
  platform::create_device_tree(dev_ids);
  cl_device_id device = dev_ids[platnum][devnum];

  vector<cl_platform_id> plat_ids;
  platform::find_platform_ids(plat_ids);
  cl_platform_id platform = plat_ids[platnum];

  cl_context ctx = platform::create_context(platform, device);
  cl_command_queue queue = platform::create_queue(ctx, device,
						  CL_QUEUE_PROFILING_ENABLE);
  
  clfftpp::clfft3 fft(nx, ny, nz, inplace, queue, ctx);

  size_t bufsize = sizeof(double) * 2 * nx * ny * nz;
  cl_int status;
  cl_mem inbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bufsize, NULL, &status);
  cl_mem outbuf;
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    outbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bufsize, NULL,&status);
  }
  
  // Create OpenCL kernel to initialize OpenCL buffer
  string init_source = "\
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
  size_t global_wsize[] = {nx, ny, nz};
  cl_program initprog = platform::create_program(init_source, ctx);
  clBuildProgram(initprog, 1, &device, NULL, NULL, NULL);
  cl_kernel initkernel = clCreateKernel(initprog, "init", &status); 
  clSetKernelArg(initkernel, 0, sizeof(cl_mem), &inbuf);

  if(N == 0) { // Transform forwards and back, outputting the buffer.
      cout << "Allocating " << 2 * nx * ny * nz << " doubles." << endl;
      double *X = new double[2 * nx * ny * nz];
      double *FX = new double[2 * nx * ny * nz];
  
    tolerance *= 1.0 + log((double) max(max(nx, ny), nz));
    cout << "Tolerance: " << tolerance << endl;

    cout << "\nInput:" << endl;
    clEnqueueNDRangeKernel(queue, initkernel, 3, NULL,  global_wsize, NULL,
			   0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, bufsize, X, 0, 0, 0);
    clFinish(queue);
    if(nx * ny * nz <= maxout) 
      show3C(X, nx, ny, nz);
    else 
      cout << X[0] << endl;

    cout << "\nTransformed:" << endl;
    fft.forward(&inbuf, inplace ? NULL : &outbuf, 0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inplace ? inbuf : outbuf, CL_TRUE, 0,
			bufsize, FX, 0, 0, 0);
    clFinish(queue);
    if(nx * ny * nz <= maxout) 
      show3C(FX, nx, ny, nz);
    else 
      cout << X[0] << endl;

    cout << "\nTransformed back:" << endl;
    fft.backward(inplace ? &inbuf : &outbuf, inplace ? NULL : &outbuf, 0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inplace ? inbuf : outbuf, CL_TRUE, 0,
			bufsize, X, 0, 0, 0);
    clFinish(queue);
    if(nx * ny * nz <= maxout) 
      show3C(X, nx, ny, nz);
    else 
      cout << X[0] << endl;
    
    // Compute the round-trip error.
    {
      double *X0 = new double[2 * nx * ny * nz];
      clEnqueueNDRangeKernel(queue, initkernel, 3, NULL,  global_wsize, NULL,
			     0, 0, 0);
      clFinish(queue);
      clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, bufsize, X0, 0, 0, 0);
      clFinish(queue);

      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < nx * ny; ++i) {
	double rdiff = X[2 * i] - X0[2 * i];
	double idiff = X[2 * i + 1] - X0[2 * i + 1];
	double diff = sqrt(rdiff * rdiff + idiff * idiff);
	L2error += diff * diff;
	if(diff > maxerror)
	  maxerror = diff;
      }
      L2error = sqrt(L2error / (double) nx);

      cout << endl;
      cout << "Round-trip error:"  << endl;
      cout << "L2 error: " << L2error << endl;
      cout << "max error: " << maxerror << endl;

      if(L2error < tolerance && maxerror < tolerance) 
	cout << "\nResults ok!" << endl;
      else {
	cout << "\nERROR: results diverge!" << endl;
	error += 1;
      }
    }

    // Compute the error with respect to FFTW
    { 
      size_t align = sizeof(Complex);
      Array::array3<Complex> f(nx, ny, nz, align);
      fftwpp::fft3d Forward(-1, f);
      fftwpp::fft3d Backward(1, f);
      double *df = (double *)f();

      clEnqueueNDRangeKernel(queue, initkernel, 3, NULL,  global_wsize, NULL,
			     0, 0, 0);
      clFinish(queue);
      clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, bufsize, df, 0, 0, 0);
      clFinish(queue);

      Forward.fft(f);

      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < nx * ny; ++i) {
    	double rdiff = FX[2 * i] - df[2 * i];
    	double idiff = FX[2 * i + 1] - df[2 * i + 1];
    	double diff = sqrt(rdiff * rdiff + idiff * idiff);
    	L2error += diff * diff;
    	if(diff > maxerror)
    	  maxerror = diff;
      }
      L2error = sqrt(L2error / (double) nx);

      cout << endl;
      cout << "Error with respect to FFTW:"  << endl;
      cout << "L2 error: " << L2error << endl;
      cout << "max error: " << maxerror << endl;

      if(L2error < tolerance && maxerror < tolerance) 
	cout << "\nResults ok!" << endl;
      else {
	cout << "\nERROR: results diverge!" << endl;
	error += 1;
      }
    }
    
    delete[] X;
    delete[] FX;

  } else { // Perform timing tests.
    double *T = new double[N];
  
    cl_ulong time_start, time_end;
    for(unsigned int i = 0; i < N; i++) {
      cl_event clv_forward;
      clEnqueueNDRangeKernel(queue, initkernel, 3, NULL,  global_wsize, NULL,
			     0, 0, 0);
      clFinish(queue);
      fft.forward(&inbuf, inplace ? NULL : &outbuf,  0, 0, &clv_forward);
      clWaitForEvents(1, &clv_forward);

      clGetEventProfilingInfo(clv_forward,
    			      CL_PROFILING_COMMAND_START,
    			      sizeof(time_start),
    			      &time_start, NULL);
      clGetEventProfilingInfo(clv_forward,
    			      CL_PROFILING_COMMAND_END,
    			      sizeof(time_end), 
    			      &time_end, NULL);
      T[i] = 1e-9 * (time_end - time_start);
    }
    timings("fft timing", nx, T, N,stats);
    delete[] T;
  }


  clReleaseMemObject(inbuf);
  if(!inplace)
    clReleaseMemObject(outbuf);
    
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return error;
}
  
