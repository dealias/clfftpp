#include <stdlib.h>
#include <platform.hpp>
#include <clfft.hpp>

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
  unsigned int N = 0;

  unsigned int maxout = 1000;
  unsigned int stats = 0; // Type of statistics used in timing test.

  double tolerance = 1e-9;
  
  int error = 0;

#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"P:D:m:x:y:N:S:hi:");
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
    case 'm':
      nx = atoi(optarg);
      ny = atoi(optarg);
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
      usage(2);
      exit(0);
      break;
    default:
      cout << "Invalid option" << endl;
      usage(2);
      exit(1);
    }
  }

  show_devices();
  cout << "Using platform " << platnum
	    << " device " << devnum 
	    << "." << endl;
  
  vector<vector<cl_device_id> > dev_ids;
  create_device_tree(dev_ids);
  cl_device_id device = dev_ids[platnum][devnum];

  vector<cl_platform_id> plat_ids;
  find_platform_ids(plat_ids);
  cl_platform_id platform = plat_ids[platnum];

  cl_context ctx = create_context(platform, device);
  cl_command_queue queue = create_queue(ctx, device, 
					CL_QUEUE_PROFILING_ENABLE);

  clfft2 fft(nx, ny, inplace, queue, ctx);
  cl_mem inbuf, outbuf;
  fft.create_cbuf(&inbuf);
  if(inplace) {
    cout << "in-place transform" << endl;
  } else {
    cout << "out-of-place transform" << endl;
    fft.create_cbuf(&outbuf);
  }

  string init_source ="\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n	\
__kernel void init(__global double *X, const unsigned int ny)		\
{						\
  const int i = get_global_id(0);		\
  const int j = get_global_id(1);		\
  unsigned pos = 2 * (i * ny + j);		\
  X[pos] = i;					\
  X[pos + 1] = j;				\
}";
  size_t global_wsize[] = {nx, ny};
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  set_kernel_arg(initkernel, 0, sizeof(cl_mem), &inbuf);
  set_kernel_arg(initkernel, 1, sizeof(unsigned int), &ny);

  cout << "Allocating " << 2 * nx * ny << " doubles." << endl;
  double *X = new double[ 2 * nx * ny];
  double *FX = new double[ 2 * nx * ny];

  if (N == 0) { // Transform forwards and back, outputting the buffer.
    tolerance *= 1.0 + log((double) max(nx, ny));
    cout << "Tolerance: " << tolerance << endl;

    cout << "\nInput:" << endl;
    clEnqueueNDRangeKernel(queue, initkernel, 2, NULL,  global_wsize, NULL,
			   0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			sizeof(double) * 2 * nx * ny, X, 0, 0, 0);
    clFinish(queue);
    if(nx * ny <= maxout) 
      show2C(X, nx, ny);
    else 
      cout << X[0] << endl;

    cout << "\nTransformed:" << endl;
    fft.forward(&inbuf, inplace ? NULL : &outbuf, 0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			sizeof(double) * 2 * nx * ny, FX, 0, 0, 0);
    clFinish(queue);
    
    if(nx * ny <= maxout) 
      show2C(FX, nx, ny);
    else 
      cout << X[0] << endl;
    
    cout << "\nTransformed back:" << endl;
    fft.backward(inplace ? &inbuf : &outbuf, inplace ? NULL : &inbuf, 0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			sizeof(double) * 2 * nx * ny, X, 0, 0, 0);
    clFinish(queue);
    
    if(nx * ny <= maxout)
      show2C(X, nx, ny);
    else 
      cout << X[0] << endl;
    
    // Compute the round-trip error.
    {
      double *X0 = new double[2 * fft.ncomplex()];
      clEnqueueNDRangeKernel(queue, initkernel, 2, NULL,  global_wsize, NULL,
			     0, 0, 0);
      clFinish(queue);
      clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			  sizeof(double) * 2 * nx * ny, X0, 0, 0, 0);
      clFinish(queue);

      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < fft.ncomplex(); ++i) {
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

      delete[] X0;
    }

    // Compute the error with respect to FFTW
    { 
      size_t align = sizeof(Complex);
      Array::array2<Complex> f(nx, ny, align);
      fftwpp::fft2d Forward(-1, f);
      fftwpp::fft2d Backward(1, f);
      double *df = (double *)f();

      clEnqueueNDRangeKernel(queue, initkernel, 2, NULL,  global_wsize, NULL,
			     0, 0, 0);
      clFinish(queue);
      clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			  sizeof(double) * 2 * nx * ny, df, 0, 0, 0);
      clFinish(queue);

      cout << "f:" << f << endl;
      Forward.fft(f);
      cout << "f:" << f << endl;

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
    delete[] FX;
    
  } else { // Perform timing tests.

    double *T = new double[N];
  
    cl_ulong time_start, time_end;
    for(unsigned int i = 0; i < N; i++) {
      cl_event clv_forward;
      clEnqueueNDRangeKernel(queue, initkernel, 2, NULL,  global_wsize, NULL,
			     0, 0, 0);
      clFinish(queue);

      
      fft.forward(&inbuf, inplace ? NULL : &outbuf, 0, 0, &clv_forward);
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

  delete X;

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return error;
}
  
