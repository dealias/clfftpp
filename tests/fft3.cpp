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
void init(T *X, unsigned int nx, unsigned int ny, unsigned int nz)
{
  for(unsigned int i = 0; i < nx; ++i) {
    for(unsigned int j = 0; j < ny; ++j) {
      for(unsigned int k = 0; k < ny; ++k) {
	  int pos = (i * ny + j) * ny + k;
	  X[2 * pos] = i;
	  X[2 * pos + 1] = j;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int platnum = 0;
  int devnum = 0;

  bool time_copy = false;
  
  bool inplace = true;

  unsigned int nx = 4;
  unsigned int ny = 4;
  unsigned int nz = 4;
  //nx=262144;

  unsigned int N = 0;

  unsigned int maxout = 10000;

  unsigned int stats = 0; // Type of statistics used in timing test.

#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"p:d:c:m:x:y:z:N:S:hi:");
    if (c == -1) break;
    
    switch (c) {
    case 'p':
      platnum = atoi(optarg);
      break;
    case 'd':
      devnum = atoi(optarg);
      break;
    case 'c':
      if(atoi(optarg) == 0)
	time_copy = false;
      else
	time_copy = true;
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
      usage(1);
      exit(0);
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
  
  clfft3 fft(nx, ny, nz, inplace, queue, ctx);
  cl_mem inbuf, outbuf;
  fft.create_cbuf(&inbuf);
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    fft.create_cbuf(&outbuf);
  }

  std::cout << "Allocating " 
	    << fft.ncomplex() 
	    << " doubles." << std::endl;
  double *X = new double[2 * fft.ncomplex()];
  double *Xout = new double[2 * fft.ncomplex()];

  cl_event r2c_event = clCreateUserEvent(ctx, NULL);
  cl_event c2r_event = clCreateUserEvent(ctx, NULL);
  cl_event forward_event = clCreateUserEvent(ctx, NULL);
  cl_event backward_event = clCreateUserEvent(ctx, NULL);
  if(N == 0) { // Transform forwards and back, outputting the buffer.
    init(X, nx, ny, nz);
    
    std::cout << "\nInput:" << std::endl;
    if(nx * ny <= maxout) 
      show3C(X, nx, ny, nz);
    else 
      std::cout << X[0] << std::endl;

    fft.ram_to_cbuf(X, &inbuf, 0, NULL, &r2c_event);
    if(inplace) {
      fft.forward(&inbuf, NULL, 1, &r2c_event, &forward_event);
      fft.cbuf_to_ram(Xout, &inbuf, 1, &forward_event, &r2c_event);
    } else {
      fft.forward(&inbuf, &outbuf, 1, &r2c_event, &forward_event);
      fft.cbuf_to_ram(Xout, &outbuf, 1, &forward_event, &r2c_event);
    }
    clWaitForEvents(1, &r2c_event);
    
    std::cout << "\nTransformed:" << std::endl;
    if(nx * ny <= maxout) 
      show3C(Xout, nx, ny, nz);
    else 
      std::cout << X[0] << std::endl;

    if(inplace) {
      fft.backward(&inbuf, NULL, 1, &forward_event, &backward_event);
    } else {
      fft.backward(&outbuf, &inbuf, 1, &forward_event, &backward_event);
    }
    fft.cbuf_to_ram(X, &inbuf, 1, &backward_event, &c2r_event);
    clWaitForEvents(1, &c2r_event);

    std::cout << "\nTransformed back:" << std::endl;
    if(nx * ny <= maxout) 
      show3C(X, nx, ny, nz);
    else 
      std::cout << X[0] << std::endl;
    
    // Compute the round-trip error.
    {
      double *X0 = new double[2 * fft.ncomplex()];
      init(X0, nx, ny, nz);
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

      std::cout << std::endl;
      std::cout << "Round-trip error:"  << std::endl;
      std::cout << "L2 error: " << L2error << std::endl;
      std::cout << "max error: " << maxerror << std::endl;
    }

    // Compute the error with respect to FFTW
    { 
      size_t align = sizeof(Complex);
      Array::array3<Complex> f(nx, ny, nz, align);
      fftwpp::fft3d Forward(-1, f);
      fftwpp::fft3d Backward(1, f);
      double *df = (double *)f();
      init(df, nx, ny, nz);
      //show3C(df, nx, ny, nz);
      Forward.fft(f);
      //show3C(df, nx, ny, nz);

      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < nx * ny; ++i) {
    	double rdiff = Xout[2 * i] - df[2 * i];
    	double idiff = Xout[2 * i + 1] - df[2 * i + 1];
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

  } else { // Perform timing tests.
    // double *T = new double[N];
    // cl_ulong time_start, time_end;
    // for(int i = 0; i < N; ++i) {
    //   init(X, nx, ny);
    //   fft.ram_to_input(X, &r2c_event);
    //   fft.forward(1, &r2c_event, &forward_event);
    //   fft.input_to_ram(X, 1, &forward_event, &c2r_event);
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
    //   T[i] = 1e-9 * (time_end - time_start); // milliseconds
    // }
    // if(time_copy) 
    //   timings("fft with copy", nx, T, N, stats);
    // else 
    //   timings("fft without copy", nx, T, N, stats);
    // delete[] T;
  }

  delete X;

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
