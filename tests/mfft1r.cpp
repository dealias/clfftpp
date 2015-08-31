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

template<class T>
void initR(T *X, const unsigned int nx, const unsigned int ny, 
	   const unsigned int dist = 0)
{
  int dist0 = dist == 0 ? nx : dist;
  for(unsigned int i = 0; i < nx; ++i) {
    for(unsigned int j = 0; j < ny; ++j) {
      int pos = j * dist0 + i;
      X[pos] = i + 10 * j;
    }
  }
}

int main(int argc, char *argv[]) {
  int platnum = 0;
  int devnum = 0;
  bool inplace = false;
  unsigned int nx = 4;
  unsigned int ny = 4;
  unsigned int n = 0;
  unsigned int M = 0;
  int instride = 0;
  int outstride = 0;
  int indist = 0;
  int outdist = 0;
  unsigned int N = 0;
  unsigned int stats = 0; // Type of statistics used in timing test.
  unsigned int maxout = 32; // maximum size of array output in entierety

  int error = 0;

#ifdef __GNUC__	
  optind = 0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"P:D:m:x:y:N:S:hi:n:M:s:t:d:e:");
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
      instride = atoi(optarg);
      break;
    case 't':
      outstride = atoi(optarg);
      break;
    case 'd':
      indist = atoi(optarg);
      break;
    case 'e':
      outdist = atoi(optarg);
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

  if(instride == 0) instride = 1;
  if(outstride == 0) outstride = 1;
  if(indist == 0) indist = nx;
  if(outdist == 0) outdist = nx / 2 + 1;

  if(n == 0) n = ny;
  if(M == 0) M = nx;

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
  cl_command_queue queue = create_queue(ctx, device, CL_QUEUE_PROFILING_ENABLE);
  
  clmfft1r fft(n, M, instride, outstride, indist, outdist, inplace, 
	       queue, ctx);

  std::cout << std::endl;
  cl_mem inbuf, outbuf;
  fft.create_cbuf(&inbuf);
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    fft.create_cbuf(&outbuf);
  }

   std::string init_source ="\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n	\
__kernel void init(__global double *X, const unsigned int nx)	\
{						\
  const int i = get_global_id(0);		\
  const int j = get_global_id(1);		\
  int pos = j * nx + i;			\
  X[pos] = i + 10 * j;				\
}";
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  set_kernel_arg(initkernel, 0, sizeof(cl_mem), &inbuf);
  set_kernel_arg(initkernel, 1, sizeof(unsigned int), &nx);
 
  std::cout << "nx: " << nx << std::endl;
  std::cout << "M: " << M << std::endl;
  std::cout << "instride: " << instride << std::endl;
  std::cout << "outstride: " << outstride << std::endl;
  std::cout << "indist: " << indist << std::endl;
  std::cout << "outdist: " << outdist << std::endl;

  std::cout << "\nAllocating " 
  	    << nx * ny
  	    << " doubles for real." << std::endl;
  double *X = new double[nx * ny];  
  int np = (n / 2 + 1);
  int ncomplex = 2 * np * M;
  std::cout << "Allocating "
  	    << ncomplex
  	    << " doubles for complex." << std::endl;
  double *FX = new double[ncomplex];

  cl_event clv_init = clCreateUserEvent(ctx, NULL);
  cl_event clv_toram = clCreateUserEvent(ctx, NULL);
  cl_event clv_forward = clCreateUserEvent(ctx, NULL);
  cl_event clv_backward = clCreateUserEvent(ctx, NULL);
  if(N == 0) {
    std::cout << "\nInput:" << std::endl;
    //initR(X, nx, M);
    //fft.ram_to_rbuf(X, &inbuf, 0, NULL, &clv_init);
    size_t global_wsize[] = {nx, ny};
    clEnqueueNDRangeKernel(queue,
			   initkernel,
			   2, // cl_uint work_dim,
			   NULL, // global_work_offset,
			   global_wsize, // global_work_size, 
			   NULL, // size_t *local_work_size, 
			   0, NULL, &clv_init);
    fft.rbuf_to_ram(X, &inbuf, 1, &clv_init, &clv_toram);
    clWaitForEvents(1, &clv_toram);
    if(nx <= maxout)
      show1R(X, nx, ny);
    else
      std::cout << X[0] << std::endl;

    fft.forward(&inbuf, inplace ? NULL : &outbuf, 1, &clv_init, &clv_forward);
    fft.cbuf_to_ram(FX, inplace ? &inbuf : &outbuf, 
		    1, &clv_forward, &clv_toram);
    clWaitForEvents(1, &clv_toram);

    std::cout << "\nTransformed:" << std::endl;
    if(nx <= maxout) {
      if(instride == 1 && outstride == 1) {
	show2C(FX, M, np);
      } else if (instride == (int)nx && outstride == (int)ny) {
	show2C(FX, np, M);
      } else {
	show2C(FX, np * M, 1);
      }
    } else {
      std::cout << FX[0] << std::endl;
    }    

    fft.backward(inplace ? &inbuf : & outbuf, 
		 inplace ? NULL : &inbuf, 1, &clv_forward, &clv_backward);
    fft.rbuf_to_ram(X, &inbuf, 1, &clv_backward, &clv_toram);
    clWaitForEvents(1, &clv_toram);

    std::cout << "\nTransformed back:" << std::endl;
    if(nx <= maxout) {
      show1R(X, n, M);
    } else {
      std::cout << X[0] << std::endl;
    }

    // Compute the round-trip error.
    {
      double *X0 = new double[fft.nreal()];
      initR(X0, nx, M);
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

      if(L2error < 1e-15 && maxerror < 1e-15) 
	std::cout << "\nResults ok!" << std::endl;
      else {
	std::cout << "\nERROR: results diverge!" << std::endl;
	error += 1;
      }
    }
    
    // Compute the error with respect to FFTW
    {
      // fftw::maxthreads=get_max_threads();
      size_t align = sizeof(Complex);
      Array::array2<double> f(nx, ny, align);
      Array::array2<Complex> g(nx, ny / 2 + 1, align);
      fftwpp::mrcfft1d Forward(n, M, instride, indist, f, g);
      double *df = (double *)f();
      double *dg = (double *)g();
      initR(df, nx, ny);

      std::cout << std::endl;
      std::cout << "fftw++ input:" << std::endl;
      std::cout << f << std::endl;

      Forward.fft(f, g);

      std::cout << "fftw++ transformed:" << std::endl;
      if(instride == 1 && outstride == 1) {
	show2C(dg, M, np);
      } else if (instride == (int)nx && outstride == (int)ny) {
	show2C(dg, np, M);
      } else {
	show2C(dg, np * M, 1);
      }
      //std::cout << g << std::endl;

      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int m = 0; m < M; ++m) {
      	for(unsigned int i = 0; i < nx / 2 + 1; ++i) {
      	  int pos = m * (nx /2 + 1) + i; 
      	  double rdiff = FX[2 * pos] - g[m][i].re;
      	  double idiff = FX[2 * pos + 1] - g[m][i].im;
      	  double diff = sqrt(rdiff * rdiff + idiff * idiff);
      	  L2error += diff * diff;
      	  if(diff > maxerror)
      	    maxerror = diff;
      	}
      }
      L2error = sqrt(L2error / (double) nx);

      std::cout << "Error with respect to FFTW:"  << std::endl;
      std::cout << "L2 error: " << L2error << std::endl;
      std::cout << "max error: " << maxerror << std::endl;

      if(L2error < 1e-15 && maxerror < 1e-15) 
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
      // initR(X, nx, M);
      // fft.ram_to_cbuf(X, &inbuf, 0, NULL, &clv_init);
      size_t global_wsize[] = {M, nx};
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

