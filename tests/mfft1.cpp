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

// Set the mfft parameters for FFTs in direction 0 or 1 in a 2D array
void direction_params(const unsigned int direction, 
		      const unsigned int nx, const unsigned int ny, 
		      unsigned int &M, unsigned int &n,
		      unsigned int &istride, unsigned int &ostride,
		      unsigned int &idist, unsigned int &odist) 
{
  switch(direction) {
  case 0:
    M = ny;
    n = nx;
    istride = ny;
    ostride = ny;
    idist = 1;
    odist = 1;
    break;
  default:
  case 1:
    M = nx;
    n = ny;
    istride = 1;
    ostride = 1;
    idist = ny;
    odist = ny;
    break;
  }
}

int main(int argc, char *argv[]) {
  int platnum = 0;
  int devnum = 0;
  bool inplace = true;
  unsigned int nx = 4;
  unsigned int ny = 4;
  unsigned int M = 0;
  unsigned int n = 0;
  unsigned int istride = 0;
  unsigned int ostride = 0;
  unsigned int idist = 0;
  unsigned int odist = 0;
  unsigned int N = 0;
  unsigned int stats = 0; // Type of statistics used in timing test.
  unsigned int maxout = 32; // maximum size of array output in entierety

  double tolerance = 1e-9;
  
  unsigned int direction = 1;

  int error = 0;

#ifdef __GNUC__	
  optind = 0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"P:D:m:x:y:N:S:hi:M:n:s:t:d:e:g:");
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
      nx = ny = atoi(optarg);
      break;
    case 'M':
      M = atoi(optarg);
      break;
    case 'n':
      n = atoi(optarg);
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
    case 's':
      istride = atoi(optarg);
      break;
    case 't':
      ostride = atoi(optarg);
      break;
    case 'd':
      idist = atoi(optarg);
      break;
    case 'e':
      odist = atoi(optarg);
      break;
    case 'g':
      direction = atoi(optarg);
      direction_params(direction, nx, ny, M, n, istride, ostride, idist, odist);
      break;
    case 'h':
      usage(2, true);
      exit(0);
      break;
    default:
      cout << "Invalid option" << endl;
      usage(2, true);
      exit(1);
    }
  }

  if(istride == 0)
    istride = 1;
  if(ostride == 0)
    ostride = 1;

  if(idist == 0)
    idist = nx;
  if(odist == 0)
    odist = nx;

  if(n == 0)
    n = nx;
  if(M == 0)
    M = ny;

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
  
  cout << "n: " << n << endl;
  cout << "M: " << M << endl;
  cout << "nx: " << nx << endl;
  cout << "ny: " << ny << endl;
  cout << "istride: " << istride << endl;
  cout << "idist: " << idist << endl;
  cout << "ostride: " << ostride << endl;
  cout << "odist: " << odist << endl;
  cout << "ny: " << ny << endl;
  clmfft1 fft(n, M, istride, ostride, idist, odist, inplace,
	      queue, ctx);

  cout << "Allocating " 
  	    << 2 * nx * ny
  	    << " doubles." << endl;
  const unsigned int ndouble = 2 * nx * ny;

  cl_int status;
  cl_mem inbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				sizeof(double) * 2 * nx * ny, NULL, &status);
  cl_mem outbuf;
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    outbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
				   sizeof(double) * 2 * nx * ny, NULL, &status);
  }
  
  string init_source = "\
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
  
  size_t global_wsize[] = {nx, ny};
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  clSetKernelArg(initkernel, 0, sizeof(cl_mem), &inbuf);

  if(N == 0) {
    tolerance *= 1.0 + log((double)nx);
    cout << "Tolerance: " << tolerance << endl;

    double *X = new double[ndouble];
    double *FX = new double[ndouble];
    
    cout << "\nInput:" << endl;
    clEnqueueNDRangeKernel(queue, initkernel, 2, NULL, global_wsize, 0, 0, 0,0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			sizeof(double) * 2 * nx * ny, X, 0, 0, 0);
    clFinish(queue);
    if(nx <= maxout)
      show2C(X, nx, ny);
    else
      cout << X[0] << endl;

    cout << "\nTransformed:" << endl;    
    fft.forward(&inbuf, inplace ? NULL : &outbuf, 0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			sizeof(double) * 2 * nx * ny, FX, 0, 0, 0);
    clFinish(queue);
    if(nx <= maxout)
      show1C(FX, n, M);
    else
      cout << FX[0] << endl;

    cout << "\nTransformed back:" << endl;
    fft.backward(inplace ? &inbuf : &outbuf, inplace ? NULL : &inbuf, 0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			sizeof(double) * 2 * nx * ny, X, 0, 0, 0);
    clFinish(queue);
    if(nx <= maxout)
      show2C(X, nx, ny);
    else
      cout << X[0] << endl;
    
    // Compute the round-trip error.
    {
      double *X0 = new double[2 * nx * ny];
      clEnqueueNDRangeKernel(queue, initkernel, 2, NULL, global_wsize, 0,
			     0, 0, 0);
      clFinish(queue);
      clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			  sizeof(double) * 2 * nx * ny, X0, 0, 0, 0);
      clFinish(queue);
      
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
      size_t align = sizeof(Complex);
      Array::array2<Complex> f(nx, ny, align);
      fftwpp::mfft1d Forward(n, -1, M, istride, idist, f);
      fftwpp::mfft1d Backward(n, 1, M, istride, idist, f);
      double *df = (double *)f();

      clEnqueueNDRangeKernel(queue, initkernel, 2, NULL, global_wsize, 0,
			     0, 0, 0);
      clFinish(queue);
      clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0,
			  sizeof(double) * 2 * nx * ny, df, 0, 0, 0);
      clFinish(queue);

      cout << endl;
      cout << "fftw++ input:" << endl;
      if(nx <= maxout)
	show2C(df, nx, ny);

      Forward.fft(f);
      cout << "fftw++ output:" << endl;
      if(nx <= maxout)
	show2C(df, nx, ny);

      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < nx; ++i) {
	for(unsigned int j = 0; j < ny; ++j) {
	  int pos = i * ny + j;
	  double rdiff = FX[2 * pos] - f[i][j].re;
	  double idiff = FX[2 * pos + 1] - f[i][j].im;
	  double diff = sqrt(rdiff * rdiff + idiff * idiff);
	  L2error += diff * diff;
	  if(diff > maxerror)
	    maxerror = diff;
	}
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
    delete[] X;
  } else {
    double *T = new double[N];
  
    cl_ulong time_start, time_end;
    for(unsigned int i = 0; i < N; i++) {
      cl_event clv_forward;
      clEnqueueNDRangeKernel(queue, initkernel, 2, NULL, global_wsize, 0,
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

  if(!inplace)
    clReleaseMemObject(outbuf);
  clReleaseMemObject(inbuf); 
  
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return error;
}
  
