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

// Set the mfft parameters for FFTs in direction 0 or 1 in a 2D array
void direction_params(const unsigned int direction, 
		      const unsigned int nx, const unsigned int ny,
		      const int inplace,
		      unsigned int &M, unsigned int &n,
		      unsigned int &istride, unsigned int &ostride,
		      unsigned int &idist, unsigned int &odist) 
{
  switch(direction) {
  case 0:
    n = nx;
    M = ny;
    istride = ny;
    ostride = ny;
    idist = 1;
    odist = 1;
    break;
  default:
  case 1:
    n = ny;
    M = nx;
    istride = 1;
    ostride = 1;
    idist = inplace ? 2 * (ny / 2 + 1) : ny;
    odist = ny / 2 + 1;
    break;
  }
}

int main(int argc, char *argv[]) {
  int platnum = 0;
  int devnum = 0;
  int inplace = 0;
  unsigned int nx = 4;
  unsigned int ny = 4;
  unsigned int n = 0;
  unsigned int M = 0;
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
    int c = getopt(argc,argv,"P:D:m:x:y:N:S:hi:n:M:s:t:d:e:g:");
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

  direction_params(direction, nx, ny, inplace,
		   M, n, istride, ostride, idist, odist);

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
  
  clfftpp::clmfft1r fft(n, M, istride, ostride, idist, odist, inplace,
			queue, ctx);

  cout << endl;

  int np = (n / 2 + 1);
  cout << "nx: " << nx << endl;
  cout << "ny: " << ny << endl;
  cout << "direction: " << direction << endl;
  cout << "inplace: " << inplace << endl;
  cout << "n: " << n << endl;
  cout << "np: " << np << endl;
  cout << "M: " << M << endl;
  cout << "istride: " << istride << endl;
  cout << "ostride: " << ostride << endl;
  cout << "idist: " << idist << endl;
  cout << "odist: " << odist << endl;

  unsigned int nreal = nx * ny;
  if(inplace) {
    if(direction == 0) 
      nreal = 2 * (nx / 2 + 1) * ny;
    if(direction == 1)
      nreal = nx * 2 * (ny / 2 + 1);
  }
  size_t inbufsize = nreal * sizeof(double);
  unsigned int ncomplex = np * M;
  size_t outbufsize = 2 * ncomplex * sizeof(double);
  
  unsigned int stride = (inplace && direction == 1) ? 2 * (ny / 2 + 1) : ny;
    
  cl_int ret;
  cl_mem inbuf = clCreateBuffer(ctx,
				CL_MEM_READ_WRITE,
				inbufsize,
				NULL,
				&ret);
  if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
  assert(ret == CL_SUCCESS);;

  cl_mem outbuf;
  if(inplace) {
    cout << "in-place transform" << endl;
  } else {
    cout << "out-of-place transform" << endl;
    //fft.create_cbuf(&outbuf);
    outbuf = clCreateBuffer(ctx,
    			    CL_MEM_READ_WRITE,
    			    outbufsize,
    			    NULL,
    			    &ret);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }
  
  cout << "\nAllocating "  << nreal << " doubles for real." << endl;
  double *X = new double[nreal];

  cout << "Allocating " << 2 * ncomplex << " doubles for complex." << endl;
  double *FX = new double[2 * ncomplex];

  string init_source ="\
#pragma OPENCL EXTENSION cl_khr_fp64: enable \n			\
__kernel void init(__global double *X,		\n		\
      const unsigned int nx, const unsigned int ny ) \n		\
{						\n		\
  const int i = get_global_id(0); 		\n		\
  const int j = get_global_id(1); 		\n		\
  const int stride = get_global_size(1); 	\n		\
  int pos = i * stride + j;			\n		\
  if((i < nx) && (j < ny))			\n		\
    X[pos] = j + 10.0 * i;			\n		\
  else						\n		\
    X[pos] = 0.0;				\n		\
}";

  cl_int status;
  cl_program initprog = platform::create_program(init_source, ctx);
  clBuildProgram(initprog, 1, &device, NULL, NULL, NULL);
  cl_kernel initkernel = clCreateKernel(initprog, "init", &status);
  clSetKernelArg(initkernel, 0, sizeof(cl_mem), &inbuf);
  clSetKernelArg(initkernel, 1, sizeof(unsigned int), &nx);
  clSetKernelArg(initkernel, 2, sizeof(unsigned int), &ny);
  size_t wsize[] = {nx, ny};
  if(inplace)
    wsize[direction] = 2 * (wsize[direction] / 2 + 1);

  cout << "wsize: " << wsize[0] << " " << wsize[1] << endl;
  
  if(N == 0) {
    tolerance *= 1.0 + log((double)nx);
    cout << "Tolerance: " << tolerance << endl;

    cout << "\nInput:" << endl;
    clEnqueueNDRangeKernel(queue, initkernel, 2, NULL, wsize, NULL, 0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, inbufsize, X, 0, 0, 0);
    clFinish(queue);
    if(nx <= maxout)
      show2R(X, wsize[0], wsize[1]);
    else
      cout << X[0] << endl;

    cout << "\nTransformed:" << endl;
    fft.forward(&inbuf, inplace ? NULL : &outbuf, 0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inplace ? inbuf : outbuf, CL_TRUE,
				     0, outbufsize, FX, 0, 0, 0);
    clFinish(queue);
    if(nx * ny <= maxout) {
      if(direction == 0) 
	show2C(FX, nx / 2 + 1, ny);
      if(direction == 1)
	show2C(FX, nx, ny / 2 + 1);
    } else {
      cout << FX[0] << endl;
    }    

    cout << "\nTransformed back:" << endl;
    fft.backward(inplace ? &inbuf : & outbuf, inplace ? NULL : &inbuf, 0, 0, 0);
    clFinish(queue);
    clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, inbufsize, X, 0, 0, 0);
    clFinish(queue);
    if(nx <= maxout) {
      show2R(X, wsize[0], wsize[1]);
    } else {
      cout << X[0] << endl;
    }

    // Compute the round-trip error.
    {
      double *X0 = new double[nreal];
      clEnqueueNDRangeKernel(queue, initkernel, 2, NULL, wsize, NULL, 0, 0, 0);
      clFinish(queue);
      clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, inbufsize, X0, 0, 0, 0);
      clFinish(queue);
      
      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < nx; ++i) {
	for(unsigned int j = 0; j < ny; ++j) {
	  unsigned int pos = j + i * stride;
	  //cout << pos << "\t" << X[pos] << "\t" << X0[pos] << endl;
	  double rdiff = X[pos] - X0[pos];
	  double diff = sqrt(rdiff * rdiff);
	  L2error += diff * diff;
	  if(diff > maxerror)
	    maxerror = diff;
	}
      }
      L2error = sqrt(L2error / (double) (nx * ny));

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
      unsigned int idims[2] = {nx, ny};
      if(inplace)
	idims[direction] = 2 * (idims[direction] / 2 + 1);
      
      double *pf = utils::doubleAlign(idims[0] * idims[1]);
      Array::array2<double> f(idims[0], idims[1], pf);

      unsigned int odims[2] = {nx, ny};
      odims[direction] = odims[direction] / 2 + 1;
      Complex *pg = inplace ? (Complex *)f()
	: utils::ComplexAlign(odims[0] * odims[1]);
      Array::array2<Complex> g(odims[0], odims[1], pg);
      
      fftwpp::mrcfft1d Forward(n, M, istride, ostride, idist, odist, f, g);

      clEnqueueNDRangeKernel(queue, initkernel, 2, NULL, wsize, NULL, 0, 0, 0);
      clFinish(queue);
      clEnqueueReadBuffer(queue, inbuf, CL_TRUE, 0, inbufsize, f(), 0, 0, 0);
      clFinish(queue);
      
      cout << "fftw++ input:\n" << f << endl;
      Forward.fft(f, g);
      cout << "fftw++ transformed:\n"  << g << endl;

      double L2error = 0.0;
      double maxerror = 0.0;

      for(unsigned int i = 0; i < ncomplex; ++i) {
	double FXre =  FX[2 * i];
	double FXim =  FX[2 * i + 1];
	// cout << i << "\t" << Complex(FXre, FXim) <<"\t" << pg[i] << endl;
      	double rdiff = FXre - pg[i].re;
      	double idiff = FXim - pg[i].im;
      	double diff = sqrt(rdiff * rdiff + idiff * idiff);
      	L2error += diff * diff;
      	if(diff > maxerror)
      	  maxerror = diff;
      }
      L2error = sqrt(L2error / (double) (nx * ny));

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
      clEnqueueNDRangeKernel(queue, initkernel, 2, NULL, wsize, NULL, 0, 0, 0);
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
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  if(error == 0)
    cout << "\nTest passed." << endl;
  else
    cout << "\nTest FAILED." << endl;
  
  return error;
}

