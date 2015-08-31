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
    istride = nx;
    ostride = nx;
    idist = 1;
    odist = 1;
    break;
  default:
  case 1:
    M = nx;
    n = ny;
    istride = 1;
    ostride = 1;
    idist = nx;
    odist = nx;
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
      std::cout << "Invalid option" << std::endl;
      usage(2, true);
      exit(1);
    }
  }

  if(istride == 0) istride = 1;
  if(ostride == 0) ostride = 1;

  if(idist == 0) idist = nx;
  if(odist == 0) odist = nx;

  if(n == 0) n = nx;
  if(M == 0) M = ny;

  show_devices();
  std::cout << "Using platform " << platnum
	    << " device " << devnum 
	    << "." << std::endl;
  
  std::vector<std::vector<cl_device_id> > dev_ids;
  create_device_tree(dev_ids);
  cl_device_id device = dev_ids[platnum][devnum];

  std::vector<cl_platform_id> plat_ids;
  find_platform_ids(plat_ids);
  cl_platform_id platform = plat_ids[platnum];

  cl_context ctx = create_context(platform, device);
  cl_command_queue queue = create_queue(ctx, device,
					CL_QUEUE_PROFILING_ENABLE);
  
  std::cout << "n: " << n << std::endl;
  std::cout << "M: " << M << std::endl;
  std::cout << "nx: " << nx << std::endl;
  std::cout << "ny: " << ny << std::endl;
  std::cout << "istride: " << istride << std::endl;
  std::cout << "idist: " << idist << std::endl;
  std::cout << "ostride: " << ostride << std::endl;
  std::cout << "odist: " << odist << std::endl;
  std::cout << "ny: " << ny << std::endl;
  clmfft1 fft(n, M, istride, ostride, idist, odist, inplace,
	      queue, ctx);

  std::cout << "Allocating " 
  	    << 2 * nx * ny
  	    << " doubles." << std::endl;
  double *X = new double[2 * nx * ny];
  double *FX = new double[2 * nx * ny];

  cl_mem inbuf, outbuf;
  fft.create_cbuf(&inbuf, nx * ny);
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    fft.create_cbuf(&outbuf, n * M);
  }
  
  std::string init_source ="\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n	\
__kernel void init(__global double *X, const unsigned int nx)		\
{						\
  const int i = get_global_id(0);		\
  const int j = get_global_id(1);		\
  int pos = i * nx + j;				\
  X[2 * pos] = i;				\
  X[2 * pos + 1] = j;				\
}";
  size_t global_wsize[] = {nx, ny};
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  set_kernel_arg(initkernel, 0, sizeof(cl_mem), &inbuf);
  set_kernel_arg(initkernel, 1, sizeof(unsigned int), &nx);

  cl_event clv_init = clCreateUserEvent(ctx, NULL);
  cl_event clv_toram = clCreateUserEvent(ctx, NULL);
  cl_event clv_forward = clCreateUserEvent(ctx, NULL);
  cl_event clv_backward = clCreateUserEvent(ctx, NULL);

  std::cout << "\nInput:" << std::endl;
  clEnqueueNDRangeKernel(queue,
			 initkernel,
			 2, // cl_uint work_dim,
			 NULL, // global_work_offset,
			 global_wsize, // global_work_size, 
			 NULL, // size_t *local_work_size, 
			 0, NULL, &clv_init);
  fft.buf_to_ram(X, &inbuf, 2 * n * M * sizeof(double), 
		 1, &clv_init, &clv_toram);
  clWaitForEvents(1, &clv_toram);

  if(nx <= maxout)
    show2C(X, nx, ny);
  else
    std::cout << X[0] << std::endl;
 

  if(N == 0) {
    //fft.ram_to_cbuf(X, &inbuf, 0, NULL, &clv_init);
    clEnqueueNDRangeKernel(queue,
			   initkernel,
			   2, // cl_uint work_dim,
			   NULL, // global_work_offset,
			   global_wsize, // global_work_size, 
			   NULL, // size_t *local_work_size, 
			   0, NULL, &clv_init);
    fft.forward(&inbuf, inplace ? NULL : &outbuf, 1, &clv_init, &clv_forward);
    fft.buf_to_ram(FX, inplace ? &inbuf : &outbuf, 2 * n * M * sizeof(double), 
		    1, &clv_forward, &clv_toram);
    clWaitForEvents(1, &clv_toram);

    std::cout << "\nTransformed:" << std::endl;
    if(nx <= maxout)
      show1C(FX, nx, ny);
    else
      std::cout << FX[0] << std::endl;

    fft.backward(inplace ? &inbuf : &outbuf,
		 inplace ? NULL : &inbuf, 1, &clv_forward, &clv_backward);
    fft.cbuf_to_ram(X, &inbuf, 1, &clv_backward, &clv_toram);
    clWaitForEvents(1, &clv_toram);

    std::cout << "\nTransformed back:" << std::endl;
    if(nx <= maxout)
      show2C(X, nx, ny);
    else
      std::cout << X[0] << std::endl;

    double errmax = 1e-10 * sqrt(nx);
    std::cout << "\nmax allowable error: " << errmax << std::endl;
    
    // Compute the round-trip error.
    {
      double *X0 = new double[2 * fft.ncomplex()];
      clEnqueueNDRangeKernel(queue,
			     initkernel,
			     2, // cl_uint work_dim,
			     NULL, // global_work_offset,
			     global_wsize, // global_work_size, 
			     NULL, // size_t *local_work_size, 
			     0, NULL, &clv_init);
      fft.buf_to_ram(X0, &inbuf, 2 * n * M * sizeof(double), 
		     1, &clv_init, &clv_toram);
      clWaitForEvents(1, &clv_toram);
      
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

      std::cout << std::endl;
      std::cout << "Round-trip error:"  << std::endl;
      std::cout << "L2 error: " << L2error << std::endl;
      std::cout << "max error: " << maxerror << std::endl;

      if(L2error < errmax && maxerror < errmax) 
	std::cout << "\nResults ok!" << std::endl;
      else {
	std::cout << "\nERROR: results diverge!" << std::endl;
	error += 1;
      }
    }
    
    // Compute the error with respect to FFTW
    {
      // //fftw::maxthreads=get_max_threads();
      size_t align = sizeof(Complex);
      Array::array2<Complex> f(nx, ny, align);
      fftwpp::mfft1d Forward(n, -1, M, istride, idist, f);
      fftwpp::mfft1d Backward(n, 1, M, istride, idist, f);
      double *df = (double *)f();

      clEnqueueNDRangeKernel(queue,
			     initkernel,
			     2, // cl_uint work_dim,
			     NULL, // global_work_offset,
			     global_wsize, // global_work_size, 
			     NULL, // size_t *local_work_size, 
			     0, NULL, &clv_init);
      fft.buf_to_ram(df, &inbuf, 2 * n * M * sizeof(double), 
		     1, &clv_init, &clv_toram);
      clWaitForEvents(1, &clv_toram);

      
      std::cout << std::endl;
      std::cout << "fftw++ input:" << std::endl;
      if(nx <= maxout)
	show2C(df, nx, ny);

      Forward.fft(f);
      std::cout << "fftw++ output:" << std::endl;
      if(nx <= maxout)
	show2C(df, nx, ny);

      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int m = 0; m < M; ++m) {
	for(unsigned int i = 0; i < nx; ++i) {
	  int pos = m * nx + i; 
	  double rdiff = FX[2 * pos] - f[m][i].re;
	  double idiff = FX[2 * pos + 1] - f[m][i].im;
	  double diff = sqrt(rdiff * rdiff + idiff * idiff);
	  L2error += diff * diff;
	  if(diff > maxerror)
	    maxerror = diff;
	}
      }
      L2error = sqrt(L2error / (double) nx);

      std::cout << std::endl;
      std::cout << "Error with respect to FFTW:"  << std::endl;
      std::cout << "L2 error: " << L2error << std::endl;
      std::cout << "max error: " << maxerror << std::endl;

      if(L2error < errmax && maxerror < errmax) 
	std::cout << "\nResults ok!" << std::endl;
      else {
	std::cout << "\nERROR: results diverge!" << std::endl;
	error += 1;
      }
    }
  } else {
    double *T = new double[N];
  
    cl_ulong time_start, time_end;
    for(unsigned int i = 0; i < N; i++) {
      //init(X, nx, M);
      //fft.ram_to_cbuf(X, &inbuf, 0, NULL, &clv_init);
      clEnqueueNDRangeKernel(queue,
			     initkernel,
			     2, // cl_uint work_dim,
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
  
