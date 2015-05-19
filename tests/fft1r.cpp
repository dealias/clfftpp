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
void initR(T *X, unsigned int n)
{
  for(unsigned int i = 0; i < n; ++i)
    X[i] = i;
}

int main(int argc, char *argv[]) {

  int platnum = 0;
  int devnum = 0;
  int nx = 4;
  unsigned int N = 0;
  unsigned int stats = 0; // Type of statistics used in timing test.
  bool inplace = false;

  int maxout = 32; // maximum size of array output in entirety

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
    case 'h':
      usage(1);
      exit(0);
      break;
    case 'i':
      inplace = atoi(optarg);
      break;
    default:
      std::cout << "Invalid option" << std::endl;
      usage(1);
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

  clfft1r fft(nx, inplace, queue, ctx);
  cl_mem inbuf, outbuf;
  if(inplace) {
    fft.create_cbuf(&inbuf);
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    fft.create_rbuf(&inbuf);
    fft.create_cbuf(&outbuf);
  }

  std::cout << "Allocating "
	    << fft.nreal() 
	    << " doubles for real." << std::endl;
  double *X = new double[inplace ? 2 * fft.ncomplex() : fft.nreal()];
  std::cout << "Allocating " 
	    << 2 * fft.ncomplex()
	    << " doubles for complex." << std::endl;
  double *FX = new double[2 * fft.ncomplex()];

  cl_event r2c_event = clCreateUserEvent(ctx, NULL);
  cl_event c2r_event = clCreateUserEvent(ctx, NULL);
  cl_event forward_event = clCreateUserEvent(ctx, NULL);
  cl_event backward_event = clCreateUserEvent(ctx, NULL);

  if(N == 0) {
    std::cout << "\nInput:" << std::endl;
    initR(X, nx);
    if(nx <= maxout)
      show1R(X, nx);
    else
      std::cout << X[0] << std::endl;

    std::cout << "\nTransformed:" << std::endl;
    fft.ram_to_rbuf(X, &inbuf, 0, NULL, &r2c_event);
    if(inplace) {
      fft.forward(&inbuf, NULL, 1, &r2c_event, &forward_event);
      fft.cbuf_to_ram(FX, &inbuf, 1, &forward_event, &c2r_event);
    } else {
      fft.forward(&inbuf, &outbuf, 1, &r2c_event, &forward_event);
      fft.cbuf_to_ram(FX, &outbuf, 1, &forward_event, &c2r_event);
    }
    clWaitForEvents(1, &c2r_event);

    if(nx <= maxout)
      show1C(FX, fft.ncomplex(0));
    else 
      std::cout << FX[0] << std::endl;

    std::cout << "\nTransformed back:" << std::endl;
    if(inplace) {
      fft.backward(&inbuf, NULL, 1, &forward_event, &backward_event);
    } else {
      fft.backward(&outbuf, &inbuf, 1, &forward_event, &backward_event);
    }
    fft.rbuf_to_ram(X, &inbuf, 1, &backward_event, NULL);
    clWaitForEvents(1, &c2r_event);
    fft.finish();
    
    if(nx <= maxout) 
      show1R(X, nx);
    else 
      std::cout << X[0] << std::endl;
    
    // Compute the round-trip error.
    {
      double *X0 = new double[fft.nreal()];
      initR(X0, nx);
      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < fft.nreal(); ++i) {
	double diff = X[i] - X0[i];
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
      //fftw::maxthreads=get_max_threads();
      size_t align = sizeof(Complex);
      Array::array1<double> f(nx, align);
      Array::array1<Complex> g(nx / 2 + 1, align);
      fftwpp::rcfft1d Forward(nx, f, g);
      fftwpp::crfft1d Backward(nx, g, f);
  
      double *df = (double *)f();
      initR(df, nx);
      //show1C(df, nx);
      Forward.fft(f, g);
      //show1C(df, nx);

      double L2error = 0.0;
      double maxerror = 0.0;
      for(int i = 0; i < nx / 2 + 1; ++i) {
	double rdiff = FX[2 * i] - g[i].re;
	double idiff = FX[2 * i + 1] - g[i].im;
	double diff = sqrt(rdiff * rdiff + idiff * idiff);
	L2error += diff * diff;
	if(diff > maxerror)
	  maxerror = diff;
      }
      L2error = sqrt(L2error / (double) nx);

      std::cout << std::endl;
      std::cout << "Error with respect to FFTW:"  << std::endl;
      std::cout << "L2 error: " << L2error << std::endl;
      std::cout << "max error: " << maxerror << std::endl;
    }
    
  } else {
    double *T = new double[N];
  
    cl_ulong time_start, time_end;
    for(unsigned int i = 0; i < N; i++) {
      initR(X, nx);

      fft.ram_to_rbuf(X, &inbuf, 0, NULL, &r2c_event);
      if(inplace) {
	fft.forward(&inbuf, NULL, 1, &r2c_event, &forward_event);
	fft.cbuf_to_ram(FX, &inbuf, 1, &forward_event, &c2r_event);
      } else {
	fft.forward(&inbuf, &outbuf, 1, &r2c_event, &forward_event);
	fft.cbuf_to_ram(FX, &outbuf, 1, &forward_event, &c2r_event);
      }
      clWaitForEvents(1, &c2r_event);
    
      clGetEventProfilingInfo(forward_event,
    			      CL_PROFILING_COMMAND_START,
    			      sizeof(time_start),
    			      &time_start, NULL);
      clGetEventProfilingInfo(forward_event,
    			      CL_PROFILING_COMMAND_END,
    			      sizeof(time_end), 
    			      &time_end, NULL);
      T[i] = 1e-6 * (time_end - time_start);
    }
    timings("fft timing", nx, T, N,stats);
    delete[] T;
  }
  delete[] X;
  delete[] FX;

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
