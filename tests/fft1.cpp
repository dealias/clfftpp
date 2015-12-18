#include <stdlib.h>
#include <platform.hpp>
#include <clfft.hpp>

#include <iostream>
#include <timing.h>

#include <getopt.h>
#include "utils.hpp"

#include "Array.h"
#include "Complex.h"
#include "fftw++.h"

using namespace std;

template<class T>
void init(T *X, unsigned int n)
{
  for(unsigned int i = 0; i < n; ++i) {
    X[2 * i] = i;
    X[2 * i + 1] = 0.0;
  }
}

int main(int argc, char *argv[]) 
{
  int platnum = 0;
  int devnum = 0;
  bool inplace = true;
  unsigned int nx = 4;
  unsigned int N = 0;
  
  unsigned int stats = 0; // Type of statistics used in timing test.
  unsigned int maxout = 32; // maximum size of array output in entierety

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
    case 'i':
      inplace = atoi(optarg);
      break;
    case 'h':
      usage(1);
      exit(0);
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

  clfft1 fft(nx, inplace, queue, ctx);
  cl_mem inbuf, outbuf;
  fft.create_cbuf(&inbuf);
  if(inplace) {
    cout << "in-place transform" << endl;
  } else {
    cout << "out-of-place transform" << endl;
    fft.create_cbuf(&outbuf);
  }

  // Create OpenCL kernel to initialize OpenCL buffer
  string init_source = "\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n	\
__kernel void init(__global double *X)\n	\
{\n						\
  const int i = get_global_id(0);\n		\
  X[2 * i] = i;\n				\
  X[2 * i + 1] = 0.0;\n				\
}\n";
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  set_kernel_arg(initkernel, 0, sizeof(cl_mem), &inbuf);
 
  cout << "Allocating " 
	    << 2 * fft.ncomplex() 
	    << " doubles." << endl;
  double *X = new double[2 * fft.ncomplex()];
  double *FX = new double[2 * fft.ncomplex()];
  
  cout << "\nInput:" << endl;
  init(X, nx);
  if(nx <= maxout)
    show1C(X, nx);
  else
    cout << X[0] << endl;
  
  cl_event clv_init = clCreateUserEvent(ctx, NULL);
  cl_event clv_toram = clCreateUserEvent(ctx, NULL);
  cl_event clv_forward = clCreateUserEvent(ctx, NULL);
  cl_event clv_backward = clCreateUserEvent(ctx, NULL);
  if(N == 0) {
    tolerance *= log((double)nx + 1);
    cout << "Tolerance: " << tolerance << endl;

    //fft.ram_to_cbuf(X, &inbuf, 0, NULL, &clv_init);
    size_t global_wsize[] = {nx};
    clEnqueueNDRangeKernel(queue,
			   initkernel,
			   1, // cl_uint work_dim,
			   NULL, // global_work_offset,
			   global_wsize, // global_work_size, 
			   NULL, // size_t *local_work_size, 
			   0, NULL, &clv_init);
    fft.forward(&inbuf, inplace ? NULL : &outbuf, 1, &clv_init, &clv_forward);
    fft.cbuf_to_ram(FX, inplace ? &inbuf : &outbuf, 
		    1, &clv_forward, &clv_toram);
    clWaitForEvents(1, &clv_toram);

    cout << "\nTransformed:" << endl;
    if(nx <= maxout)
      show1C(FX, nx);
    else
      cout << FX[0] << endl;
    
    fft.backward(&inbuf, inplace ? NULL : &outbuf, 
		 1, &clv_forward, &clv_backward);
    fft.cbuf_to_ram(X, &inbuf, 1, &clv_backward, &clv_toram);
    clWaitForEvents(1, &clv_toram);

    cout << "\nTransformed back:" << endl;
    if(nx <= maxout)
      show1C(X, nx);
    else
      cout << X[0] << endl;

    // Compute the round-trip error.
    {
      double *X0 = new double[2 * fft.ncomplex()];
      init(X0, nx);
      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < nx; ++i) {
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
      //fftw::maxthreads=get_max_threads();
      size_t align = sizeof(Complex);
      Array::array1<Complex> f(nx, align);
      fftwpp::fft1d Forward(-1, f);
      fftwpp::fft1d Backward(1, f);
      double *df = (double *)f();
      init(df, nx);
      //show1C(df, nx);
      Forward.fft(f);
      //show1C(df, nx);

      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < nx; ++i) {
	double rdiff = FX[2 * i] - f[i].re;
	double idiff = FX[2 * i + 1] - f[i].im;
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
      //init(X, nx);
      //fft.ram_to_cbuf(X, &inbuf, 0, NULL, &clv_init);
      set_kernel_arg(initkernel, 0, sizeof(cl_mem), &inbuf);
      size_t global_wsize[] = {nx};
      clEnqueueNDRangeKernel(queue,
			     initkernel,
			     1, // cl_uint work_dim,
			     NULL, // global_work_offset,
			     global_wsize, // global_work_size, 
			     NULL, // size_t *local_work_size, 
			     0, NULL, &clv_init);

      fft.forward(&inbuf, inplace ? NULL : &outbuf, 1, &clv_init, &clv_forward);
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
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return error;
}
  
