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
  unsigned int nz = 4;
  unsigned int N = 0;
  unsigned int stats = 0; // Type of statistics used in timing test.
  bool inplace = false;

  double tolerance = 1e-9;
  
  unsigned int maxout = 100; // maximum size of array output in entirety

  int error = 0;
#ifdef __GNUC__	
  optind = 0;
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

  unsigned int nzp = nz / 2 + 1;
  unsigned int skip = inplace ? 2 * nzp : nz;
  
  clfftpp::clfft3r fft(nx, ny, nz, inplace, queue, ctx);

  size_t rbufsize = sizeof(double) * nx * ny * skip;
  size_t cbufsize = 2 * sizeof(double) * nx * ny * nzp;
  cl_int status;
  cl_mem inbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				rbufsize, NULL,
				&status);
  cl_mem outbuf;
  if(inplace) {
    cout << "in-place transform" << endl;
  } else {
    cout << "out-of-place transform" << endl;
    outbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, cbufsize, NULL, &status);
  }

  string init_source ="\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n		\
__kernel void init(__global double *X,			\
const unsigned int skip)				\
{							\
  const int i = get_global_id(0);			\
  const int j = get_global_id(1);			\
  const int k = get_global_id(2);			\
  const int nx = get_global_size(0);			\
  const int ny = get_global_size(1);			\
  unsigned int pos = i * ny * skip + j * skip +  + k;	\
  X[pos] = i * i + j + 10 * k;				\
}";
  cl_program initprog = platform::create_program(init_source, ctx);
  clBuildProgram(initprog, 1, &device, NULL, NULL, NULL);
  cl_kernel initkernel = clCreateKernel(initprog, "init", &status); 
  clSetKernelArg(initkernel, 0, sizeof(cl_mem), &inbuf);
  clSetKernelArg(initkernel, 1, sizeof(unsigned int), &skip);
  size_t global_wsize[] = {nx, ny, nz};
    

  if(N == 0) {
    cout << "Allocating "  << nx * ny * skip << " doubles for real." << endl;
    double *X = new double[nx * ny * skip];
    cout << "Allocating "  << nx * ny * nzp << " doubles for complex." << endl;
    double *FX = new double[2 * nx * ny * nzp];
  
    tolerance *= 1.0 + log((double) max(max(nx, ny), nz));
    cout << "Tolerance: " << tolerance << endl;

    cout << "\nInput:" << endl;
    clEnqueueNDRangeKernel(queue, initkernel, 3, NULL,  global_wsize, NULL,
			   0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			rbufsize, X, 0, 0, 0);
    clFinish(queue);
    if(nx * ny * nz <= maxout)
      show3R(X, nx, ny, nz, skip);
    else
      cout << X[0] << endl;
    
    fft.forward(&inbuf, inplace ? NULL : &outbuf, 0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inplace ? inbuf : outbuf, CL_TRUE, 0,
			cbufsize, FX, 0, 0, 0);
    clFinish(queue);
    
    cout << "\nTransformed:" << endl;
    if(nx * ny * nz <= maxout) {
      show3C(FX, nx, ny, nzp);
    } else {
      cout << FX[0] << endl;
    }

    fft.backward(inplace ? &inbuf : &outbuf, inplace? NULL : &inbuf, 0, 0, 0);
    clFinish(queue);
    
    cout << "\nTransformed back:" << endl;
    if(nx <= maxout)
      show3R(X, nx, ny, nz, skip);
    else 
      cout << X[0] << endl;

    // compute the round-trip error.
    {
      double *X0 = new double[nx * ny * skip];
      clEnqueueNDRangeKernel(queue, initkernel, 3, NULL,  global_wsize, NULL,
			     0, 0, 0);
      clFinish(queue);
      clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			  rbufsize, X0, 0, 0, 0);
      clFinish(queue);
      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < nx; ++i) {
	for(unsigned int j = 0; j < ny; ++j) {
	  for(unsigned int k = 0; k < nz; ++k) {
	    unsigned int pos = i * ny * skip + j * skip +  + k;
	    double diff = fabs(X[pos] - X0[pos]);
	    L2error += diff * diff;
	    if(diff > maxerror)
	      maxerror = diff;
	  }
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
      Array::array3<double> f(nx, ny, inplace ? 2 * nzp : nz, align);
      Complex *pg = inplace ? (Complex*)f() : utils::ComplexAlign(nx * ny *nzp);
      Array::array3<Complex> g(nx, ny, nzp, pg);
      
      fftwpp::rcfft3d Forward(nx, ny, nz, f, g);
      fftwpp::crfft3d Backward(nx, ny, nz, g, f);

      clEnqueueNDRangeKernel(queue, initkernel, 3, NULL,  global_wsize, NULL,
			     0, 0, 0);
      clFinish(queue);
      clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			  rbufsize, f(), 0, 0, 0);
      clFinish(queue);

      if(nx * ny * nz <= maxout)
	cout << "fftw: f" << endl << f << endl;
      Forward.fft(f, g);
      if(nx * ny * nz <= maxout)
	cout << "fftw: g" << endl << g << endl;

      double *dg = (double*) pg;
      
      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < nx; ++i) {
      	for(unsigned int j = 0; j < ny; ++j) {
  	  for(unsigned int k = 0; k < nzp; ++k) {
	    unsigned int pos = i * ny * nzp + j * nzp +  + k;
  	    double rdiff = FX[2 * pos] - dg[2 * pos];
  	    double idiff = FX[2 * pos + 1] - dg[2 * pos + 1];
  	    double diff = sqrt(rdiff * rdiff + idiff * idiff);
  	    L2error += diff * diff;
  	    if(diff > maxerror)
  	      maxerror = diff;
  	  }
  	  //cout << endl;
  	}
      }
      L2error = sqrt(L2error / (double) (nx * ny * nzp));

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
    delete[] X;
  

  } else {
    double *T = new double[N];
  
    cl_ulong time_start, time_end;
    for(unsigned int i = 0; i < N; i++) {
      cl_event clv_forward;
      
      size_t global_wsize[] = {nx, ny, nz};
      clEnqueueNDRangeKernel(queue, initkernel, 3, NULL,  global_wsize, NULL,
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

  clReleaseMemObject(inbuf);
  if(!inplace)
    clReleaseMemObject(outbuf);
    
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return error;
}
  
