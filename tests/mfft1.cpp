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
void init(T *X, const unsigned int nx, const unsigned int M)
{
  for(unsigned int m = 0; m < M; ++m) {
    for(unsigned int i = 0; i < nx; ++i) {
      int pos = m * nx + i; 
      X[2 * pos] = i;
      X[2 * pos + 1] = 0.0;
    }
  }
}

int main(int argc, char *argv[]) {
  int platnum = 0;
  int devnum = 0;
  bool inplace = true;
  unsigned int nx = 4;
  unsigned int M = 7;
  int instride = 1;
  int outstride = 1;
  int indist = nx;
  int outdist = nx;
  int N = 0;
  unsigned int stats = 0; // Type of statistics used in timing test.
  unsigned int maxout = 32; // maximum size of array output in entierety

#ifdef __GNUC__	
  optind = 0;
#endif
  for (;;) {
    int c = getopt(argc,argv,"P:D:m:x:N:S:hi:M:s:t:d:e:");
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
    case 'M':
      M = atoi(optarg);
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
      usage(1, true);
      exit(0);
      break;
    default:
      std::cout << "Invalid option" << std::endl;
      usage(1, true);
      exit(1);
    }
  }

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
  
  clmfft1 fft(nx, M, instride, outstride, indist, outdist, inplace,
	      queue, ctx);

  cl_mem inbuf, outbuf;
  fft.create_cbuf(&inbuf);
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    fft.create_cbuf(&outbuf);
  }
  
  std::cout << "Allocating " 
  	    << 2 * fft.ncomplex() 
  	    << " doubles." << std::endl;
  double *X = new double[2 * fft.ncomplex()];
  double *FX = new double[2 * fft.ncomplex()];

  std::cout << "\nInput:" << std::endl;
  init(X, nx, M);
  if(nx <= maxout)
    show1C(X, nx, M);
  else
    std::cout << X[0] << std::endl;
 
  cl_event r2c_event = clCreateUserEvent(ctx, NULL);
  cl_event c2r_event = clCreateUserEvent(ctx, NULL);
  cl_event forward_event = clCreateUserEvent(ctx, NULL);
  cl_event backward_event = clCreateUserEvent(ctx, NULL);
   if(N == 0) {
     fft.ram_to_cbuf(X, &inbuf, 0, NULL, &r2c_event);
    if(inplace) {
      fft.forward(&inbuf, NULL, 1, &r2c_event, &forward_event);
      fft.cbuf_to_ram(FX, &inbuf, 1, &forward_event, &r2c_event);
    } else {
      fft.forward(&inbuf, &outbuf, 1, &r2c_event, &forward_event);
      fft.cbuf_to_ram(FX, &outbuf, 1, &forward_event, &r2c_event);
    }
    clWaitForEvents(1, &r2c_event);

    std::cout << "\nTransformed:" << std::endl;
    if(nx <= maxout)
      show1C(FX, nx, M);
    else
      std::cout << FX[0] << std::endl;
    
    if(inplace) {
      fft.backward(&inbuf, NULL, 1, &forward_event, &backward_event);
    } else {
      fft.backward(&outbuf, &inbuf, 1, &forward_event, &backward_event);
    }
    fft.cbuf_to_ram(X, &inbuf, 1, &backward_event, &c2r_event);
    clWaitForEvents(1, &c2r_event);

    std::cout << "\nTransformed back:" << std::endl;
    if(nx <= maxout)
      show1C(X, nx, M);
    else
      std::cout << X[0] << std::endl;

    // Compute the round-trip error.
    {
      double *X0 = new double[2 * fft.ncomplex()];
      init(X0, nx, M);
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
    }
    
    // Compute the error with respect to FFTW
    {
      // //fftw::maxthreads=get_max_threads();
      size_t align = sizeof(Complex);
      Array::array2<Complex> f(M, nx, align);
      fftwpp::mfft1d Forward(nx, -1, M, instride, indist, f);
      fftwpp::mfft1d Backward(nx, 1, M, instride, indist, f);
      double *df = (double *)f();
      init(df, nx, M);
      //show1C(df, nx, M);
      Forward.fft(f);
      // //show1C(df, nx);

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
    }

  } else {
    // double *T = new double[N];
  
    // cl_ulong time_start, time_end;
    // for(int i=0; i < N; ++i) {
    //   init(X,nx);
    //   seconds();
    //   fft.ramtoinput(X, &r2c_event);
    //   fft.forward(1, &r2c_event, &forward_event);
    //   fft.inputtoram(X, 1, &forward_event, &c2r_event);
    //   clWaitForEvents(1, &c2r_event);

    //   if(time_copy) {
    // 	clGetEventProfilingInfo(r2c_event,
    // 				CL_PROFILING_COMMAND_START,
    // 				sizeof(time_start),
    // 				&time_start, NULL);
    // 	clGetEventProfilingInfo(c2r_event,
    // 				CL_PROFILING_COMMAND_END,
    // 				sizeof(time_end), 
    // 				&time_end, NULL);
    //   } else {
    // 	clGetEventProfilingInfo(forward_event,
    // 				CL_PROFILING_COMMAND_START,
    // 				sizeof(time_start),
    // 				&time_start, NULL);
    // 	clGetEventProfilingInfo(forward_event,
    // 				CL_PROFILING_COMMAND_END,
    // 				sizeof(time_end), 
    // 				&time_end, NULL);
    //   }
    //   T[i] = 1e-6 * (time_end - time_start);
    // }
    // if(time_copy)
    //   timings("fft with copy",nx,T,N,stats);
    // else 
    //   timings("fft without copy",nx,T,N,stats);
    // delete[] T;
  }

  delete[] X;
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
