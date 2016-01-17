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
#include "align.h"

using namespace std;

int main(int argc, char *argv[]) {
  int platnum = 0;
  int devnum = 0;
  unsigned int nx = 4;
  unsigned int ny = 4;
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

  unsigned int nyp = ny / 2 + 1;
  int skip = inplace ? 2 * nyp : ny;
  
  clfftpp::clfft2r fft(nx, ny, inplace, queue, ctx);

  cl_int status;

  cl_mem inbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				sizeof(double) * nx * skip, NULL,
				&status);
  cl_mem outbuf;
  if(inplace) {
    cout << "in-place transform" << endl;
  } else {
    cout << "out-of-place transform" << endl;
    outbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			    2 * sizeof(double) * nx * nyp, NULL, &status);
  }

  string init_source ="\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n	\
__kernel void init(__global double *X, const unsigned int skip)	\
{						\
  const int i = get_global_id(0);		\
  const int j = get_global_id(1);		\
  unsigned pos = i * skip + j;			\
  X[pos] = i + j;			\
}";
  
  size_t global_wsize[] = {nx, ny};
  cl_program initprog = platform::create_program(init_source, ctx);
  clBuildProgram(initprog, 1, &device, NULL, NULL, NULL);
  cl_kernel initkernel = clCreateKernel(initprog, "init", &status); 
  clSetKernelArg(initkernel, 0, sizeof(cl_mem), &inbuf);
  clSetKernelArg(initkernel, 1, sizeof(unsigned int), &skip);

  cout << "Allocating " << nx * skip  << " doubles for real." << endl;
  double *X = new double[nx * skip];
  cout << "Allocating "  << 2 * nx * nyp << " doubles for complex." << endl;
  double *FX = new double[2 * nx * nyp];

  if(N == 0) {
    tolerance *= 1.0 + log((double) max(nx, ny));
    cout << "Tolerance: " << tolerance << endl;

    cout << "\nInput:" << endl;
    clEnqueueNDRangeKernel(queue, initkernel, 2, NULL,  global_wsize, NULL,
			   0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			sizeof(double) * nx * skip, X, 0, 0, 0);
    clFinish(queue);
    
    if(nx <= maxout)
      show2R(X, nx, ny, skip);
    else
      cout << X[0] << endl;
    
    fft.forward(&inbuf, inplace ? NULL : &outbuf, 0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inplace ? inbuf : outbuf, CL_TRUE, 0,
			2 * sizeof(double) * nx * nyp, FX, 0, 0, 0);
    clFinish(queue);
    
    cout << "\nTransformed:" << endl;
    if(nx <= maxout) {
      show2C(FX, nx, nyp);
    } else {
      cout << FX[0] << endl;
    }
    
    fft.backward(inplace ? &inbuf : &outbuf, 
		 inplace ? NULL : &inbuf, 0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			sizeof(double) * nx * skip, X, 0, 0, 0);
    clFinish(queue);

    cout << "\nTransformed back:" << endl;
    if(nx <= maxout)
      show2R(X, nx, ny, skip);
    else 
      cout << X[0] << endl;

    // compute the round-trip error.
    {
      double *X0 = new double[nx * skip];
     
      clEnqueueNDRangeKernel(queue, initkernel, 2, NULL,  global_wsize, NULL,
			     0, 0, 0);
      clFinish(queue);
      clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			  sizeof(double) * nx * skip, X0, 0, 0, 0);
      clFinish(queue);

      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < nx; ++i) {
	for(unsigned int j = 0; j < ny; ++j) {
	  unsigned pos = i * skip + j;
	  double diff = fabs(X[pos] - X0[pos]);
	  L2error += diff * diff;
	  if(diff > maxerror)
    	  maxerror = diff;
	}
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
      fftwpp::fftw::maxthreads = get_max_threads();
      size_t align = sizeof(Complex);
      Array::array2<double> f(nx, inplace ? 2 * nyp : ny, align);
      Complex *pg = inplace ? (Complex*)f() : utils::ComplexAlign(nx * nyp);
      Array::array2<Complex> g(nx, nyp, pg); 
     
      fftwpp::rcfft2d Forward(nx, ny, f, g);
      fftwpp::crfft2d Backward(nx, ny, g, f);
    
      clEnqueueNDRangeKernel(queue, initkernel, 2, NULL,  global_wsize, NULL,
			     0, 0, 0);
      clFinish(queue);
      clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			  sizeof(double) * nx * skip, (double *)f(), 0, 0, 0);
      clFinish(queue);

      cout << f << endl;
      Forward.fft(f, g);
      cout << g << endl;

      double *dg = (double*) pg;
      
      double L2error = 0.0;
      double maxerror = 0.0;
      unsigned int ndcomplex = nx * nyp;
      for(unsigned int i = 0; i < ndcomplex; i += 2) {
	double rdiff = FX[i] - dg[i];
	double idiff = FX[i + 1] - dg[i + 1];
	cout << Complex(FX[i],FX[i + 1]) << "\t"
	     << Complex(dg[i],dg[i + 1]) << endl;
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
      
      size_t global_wsize[] = {nx, ny};
      clEnqueueNDRangeKernel(queue,
			     initkernel,
			     2, // cl_uint work_dim,
			     NULL, // global_work_offset,
			     global_wsize, // global_work_size, 
			     NULL, // size_t *local_work_size, 
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
  delete[] FX;
  delete[] X;
  
  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return error;
}
  
