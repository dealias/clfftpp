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

template<class T>
void init(T *X, unsigned int nx, unsigned int ny)
{
  for(unsigned int i = 0; i < nx; ++i) {
    for(unsigned int j = 0; j < ny; ++j) {
      unsigned pos = 2 * (i * ny + j); 
      X[pos] = i;
      X[pos + 1] = j;
    }
  }
}

int main(int argc, char *argv[]) {
  int platnum = 0;
  int devnum = 0;
  bool inplace = true;
  unsigned int nx = 4;
  unsigned int ny = 4;
  unsigned int N = 0;

  unsigned int maxout = 10000;
  unsigned int stats = 0; // Type of statistics used in timing test.

  double tolerance = 1e-10;
  
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
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  set_kernel_arg(initkernel, 0, sizeof(cl_mem), &inbuf);
  set_kernel_arg(initkernel, 1, sizeof(unsigned int), &ny);

  cout << "Allocating " << fft.ncomplex() << " doubles." << endl;
  double *X = new double[2 * fft.ncomplex()];
  double *FX = new double[2 * fft.ncomplex()];

  cl_event clv_init = clCreateUserEvent(ctx, NULL);
  cl_event clv_toram = clCreateUserEvent(ctx, NULL);
  cl_event clv_forward = clCreateUserEvent(ctx, NULL);
  cl_event clv_backward = clCreateUserEvent(ctx, NULL);

  if (N == 0) { // Transform forwards and back, outputting the buffer.
    tolerance *= log((double) max(nx, ny));
    cout << "Tolerance: " << tolerance << endl;

    init(X, nx, ny);
    
    cout << "\nInput:" << endl;
    if(nx * ny <= maxout) 
      show2C(X, nx, ny);
    else 
      cout << X[0] << endl;

    //fft.ram_to_cbuf(X, &inbuf, 0, NULL, &clv_init);
    size_t global_wsize[] = {nx, ny};
    clEnqueueNDRangeKernel(queue,
			   initkernel,
			   2, // cl_uint work_dim,
			   NULL, // global_work_offset,
			   global_wsize, // global_work_size, 
			   NULL, // size_t *local_work_size, 
			   0, NULL, &clv_init);

    fft.forward(&inbuf, inplace ? NULL : &outbuf, 
		1, &clv_init, &clv_forward);
    fft.cbuf_to_ram(FX, inplace ? &inbuf : &outbuf, 
		    1, &clv_forward, &clv_toram);
    clWaitForEvents(1, &clv_toram);
    
    cout << "\nTransformed:" << endl;
    if(nx * ny <= maxout) 
      show2C(FX, nx, ny);
    else 
      cout << X[0] << endl;

    fft.backward(inplace ? &inbuf : &outbuf, 
		 inplace ? NULL : &inbuf, 
		 1, &clv_forward, &clv_backward);
    fft.cbuf_to_ram(X, &inbuf, 1, &clv_backward, &clv_toram);
    clWaitForEvents(1, &clv_toram);

    cout << "\nTransformed back:" << endl;
    if(nx * ny <= maxout) 
      show2C(X, nx, ny);
    else 
      cout << X[0] << endl;
    
    // Compute the round-trip error.
    {
      double *X0 = new double[2 * fft.ncomplex()];
      init(X0, nx, ny);
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
    }

    // Compute the error with respect to FFTW
    { 
      size_t align = sizeof(Complex);
      Array::array2<Complex> f(nx, ny, align);
      fftwpp::fft2d Forward(-1, f);
      fftwpp::fft2d Backward(1, f);
      double *df = (double *)f();
      init(df, nx, ny);
      //show2C(df, nx, ny);
      Forward.fft(f);
      //show2C(df, nx, ny);

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

  } else { // Perform timing tests.

    double *T = new double[N];
  
    cl_ulong time_start, time_end;
    for(unsigned int i = 0; i < N; i++) {
      //init(X, nx, ny);
      //fft.ram_to_cbuf(X, &inbuf, 0, NULL, &clv_init);
      
      size_t global_wsize[] = {nx, ny};
      clEnqueueNDRangeKernel(queue,
			     initkernel,
			     2, // cl_uint work_dim,
			     NULL, // global_work_offset,
			     global_wsize, // global_work_size, 
			     NULL, // size_t *local_work_size, 
			     0, NULL, &clv_init);

      fft.forward(&inbuf, inplace ? NULL : &outbuf, 
		  1, &clv_init, &clv_forward);
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
  
