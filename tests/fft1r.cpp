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
  unsigned int nx = 4;
  unsigned int N = 0;
  unsigned int stats = 0; // Type of statistics used in timing test.
  bool inplace = false;

  unsigned int maxout = 32; // maximum size of array output in entirety

  double tolerance = 1e-9;
  
  int error = 0;

#ifdef __GNUC__	
  optind = 0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"P:D:m:x:N:S:hi:");
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
    case 'm':
      nx = atoi(optarg);
      break;
    case 'N':
      N = atoi(optarg);
      break;
    case 'S':
      stats = atoi(optarg);
      break;
    case 'h':
      usage(1);
      exit(0);
      break;
    case 'i':
      inplace = atoi(optarg);
      break;
    default:
      cout << "Invalid option" << endl;
      usage(1);
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
  cl_command_queue queue = create_queue(ctx, device, CL_QUEUE_PROFILING_ENABLE);

  clfft1r fft(nx, inplace, queue, ctx);

  unsigned int nxp = nx / 2 + 1;
  unsigned int nreal = inplace ? 2 * nxp : nx; 

  cl_int status;
  cl_mem inbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				sizeof(double) * nreal, NULL,
				&status);
  cl_mem outbuf;
  if(inplace) {
    cout << "in-place transform" << endl;
  } else {
    cout << "out-of-place transform" << endl;
    outbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			   sizeof(double) * 2 * nxp, NULL,
			   &status);
  }

  // Create OpenCL kernel to initialize OpenCL buffer
  size_t global_wsize[] = {nx};
  string init_source = "\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n	\
__kernel void init(__global double *X)\n	\
{\n						\
  const int i = get_global_id(0);\n		\
  X[i] = i;\n					\
}\n";
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  clSetKernelArg(initkernel, 0, sizeof(cl_mem), &inbuf);

  cout << "Allocating " << nreal  << " doubles for real." << endl;
  double *X = new double[nreal];
  cout << "Allocating " << 2 * nxp  << " doubles for complex." << endl;
  double *FX = new double[2 * nxp];

  if(N == 0) {
    tolerance *= 1.0 + log((double)nx);
    cout << "Tolerance: " << tolerance << endl;

    cout << "\nInput:" << endl;
    clEnqueueNDRangeKernel(queue, initkernel, 1, NULL, global_wsize, 0, 0, 0,0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			sizeof(double) * nx, X, 0, 0, 0);
    clFinish(queue);
    if(nx <= maxout)
      show1R(X, nx);
    else
      cout << X[0] << endl;

    cout << "\nTransformed:" << endl;
    fft.forward(&inbuf, inplace ? NULL : &outbuf, 0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inplace ? inbuf : outbuf, CL_TRUE, 0,
			sizeof(double) * 2 * nxp, FX, 0, 0, 0);
    clFinish(queue);
    if(nx <= maxout)
      show1C(FX, nxp);
    else 
      cout << FX[0] << endl;
    
    cout << "\nTransformed back:" << endl;
    fft.backward(inplace ? &inbuf : &outbuf, inplace ? NULL : &inbuf, 0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			sizeof(double) * nreal, X, 0, 0, 0);
    clFinish(queue);

    if(nx <= maxout) 
      show1R(X, nx);
    else 
      cout << X[0] << endl;
    
    // Compute the round-trip error.
    {
      double *X0 = new double[nreal];
      clEnqueueNDRangeKernel(queue, initkernel, 1, NULL, global_wsize,
			     0, 0, 0, 0);
      clFinish(queue);
      clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			  sizeof(double) * nx, X0, 0, 0, 0);
      clFinish(queue);

      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < nx; ++i) {
  	double diff = X[i] - X0[i];
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

      free(X0);
    }

    // Compute the error with respect to FFTW
    {
      //fftw::maxthreads=get_max_threads();
      size_t align = sizeof(Complex);
      Array::array1<double> f(nx, align);
      Array::array1<Complex> g(nx / 2 + 1, align);
      fftwpp::rcfft1d Forward(nx, f, g);
      fftwpp::crfft1d Backward(nx, g, f);
  
      double *df = (double *)f();
      clEnqueueNDRangeKernel(queue, initkernel, 1, NULL, global_wsize,
			     0, 0, 0, 0);
      clFinish(queue);
      clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			  sizeof(double) * nx, df, 0, 0, 0);
      clFinish(queue);

      if(nx <= maxout)
	cout << "f:" << f << endl;
      Forward.fft(f, g);
      if(nx <= maxout)
	cout << "g:" << g << endl;

      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < nx / 2 + 1; ++i) {
  	double rdiff = FX[2 * i] - g[i].re;
  	double idiff = FX[2 * i + 1] - g[i].im;
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
    
  } else {
    double *T = new double[N];
  
    cl_ulong time_start, time_end;
    for(unsigned int i = 0; i < N; i++) {
      cl_event clv_forward;
      clEnqueueNDRangeKernel(queue, initkernel, 1, NULL, global_wsize,
			     0, 0, 0, 0);
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
  delete[] X;
  delete[] FX;

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return error;
}
  
