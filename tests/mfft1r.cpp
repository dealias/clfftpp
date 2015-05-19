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
void initR(T *X, const unsigned int nx, const unsigned int M, 
	   const unsigned int dist = 0)
{
  int dist0 = dist == 0 ? nx : dist;
  for(unsigned int m = 0; m < M; ++m) {
    for(unsigned int i = 0; i < nx; ++i) {
      int pos = m * dist0 + i;
      X[pos] = i + 10 * m;
    }
  }
}

int main(int argc, char *argv[]) {
  int platnum = 0;
  int devnum = 0;
  bool inplace = false;
  unsigned int nx = 4;
  unsigned int M = 4;
  int istride = 1;
  int ostride = 1;
  int idist = nx;
  int odist = nx / 2 + 1;
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
  cl_command_queue queue = create_queue(ctx, device, CL_QUEUE_PROFILING_ENABLE);
  
  clmfft1r fft(nx, M, istride, ostride, idist, odist, inplace, queue, ctx);

  std::cout << std::endl;
  cl_mem inbuf, outbuf;
  fft.create_cbuf(&inbuf);
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    fft.create_cbuf(&outbuf);
  }
  std::cout << "nx: " << nx << std::endl;
  std::cout << "M: " << M << std::endl;
  std::cout << "istride: " << istride << std::endl;
  std::cout << "ostride: " << ostride << std::endl;
  std::cout << "idist: " << idist << std::endl;
  std::cout << "odist: " << odist << std::endl;

  std::cout << "\nAllocating " 
  	    << fft.nreal() 
  	    << " doubles for real." << std::endl;
  double *X = new double[fft.nreal()];  
  std::cout << "Allocating " 
  	    << 2 * fft.ncomplex() 
  	    << " doubles for complex." << std::endl;
  double *FX = new double[2 * fft.ncomplex()];

  std::cout << "\nInput:" << std::endl;
  initR(X, nx, M);
  if(nx <= maxout) {
    //show1R(X, nx, M);
    show1R(X, nx * M, 1);
  } else {
    std::cout << X[0] << std::endl;
  } 

  cl_event r2c_event = clCreateUserEvent(ctx, NULL);
  cl_event c2r_event = clCreateUserEvent(ctx, NULL);
  cl_event forward_event = clCreateUserEvent(ctx, NULL);
  cl_event backward_event = clCreateUserEvent(ctx, NULL);
  if(N == 0) {
    fft.ram_to_rbuf(X, &inbuf, 0, NULL, &r2c_event);
    if(inplace) {
      fft.forward(&inbuf, NULL, 1, &r2c_event, &forward_event);
      fft.cbuf_to_ram(FX, &inbuf, 1, &forward_event, &r2c_event);
    } else {
      fft.forward(&inbuf, &outbuf, 1, &r2c_event, &forward_event);
      fft.cbuf_to_ram(FX, &outbuf, 1, &forward_event, &r2c_event);
    }
    clWaitForEvents(1, &r2c_event);

    std::cout << "\nTransformed:" << std::endl;
    if(nx <= maxout) {
      show2C(FX, 1, M * (nx / 2 + 1));
    } else {
      std::cout << FX[0] << std::endl;
    }    

    if(inplace) {
      fft.backward(&inbuf, NULL, 1, &forward_event, &backward_event);
    } else {
      fft.backward(&outbuf, &inbuf, 1, &forward_event, &backward_event);
    }
    fft.rbuf_to_ram(X, &inbuf, 1, &backward_event, &c2r_event);

    clWaitForEvents(1, &c2r_event);

    std::cout << "\nTransformed back:" << std::endl;
    if(nx <= maxout) {
      //show1R(X, nx, M);
      show1R(X, nx * M, 1);
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
    }
    
    // Compute the error with respect to FFTW
    {
      // //fftw::maxthreads=get_max_threads();
      size_t align = sizeof(Complex);
      Array::array2<double> f(M, nx, align);
      Array::array2<Complex> g(M, nx / 2 + 1, align);
      fftwpp::mrcfft1d Forward(nx, M, istride, idist, f, g);
      //fftwpp::mcrfft1d Backward(nx, M, ostride, odist, g, f);
      double *df = (double *)f();
      double *dg = (double *)g();
      initR(df, nx, M);
      Forward.fft(f, g);

      show2C(dg, 1, M * (nx / 2 + 1));
      std::cout << g << std::endl;

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
  
