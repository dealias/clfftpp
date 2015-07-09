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
      for(unsigned int k = 0; k < nz; ++k) {
	int pos = i * nz * ny + j * nz + k;
	X[2 * pos] = i;
	X[2 * pos + 1] = j + k * k;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int platnum = 0;
  int devnum = 0;

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
      std::cout << "Invalid option" << std::endl;
      usage(3);
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

    // Create OpenCL kernel to initialize OpenCL buffer
  std::string init_source = "\
__kernel void init(__global double *X,		\
const unsigned int ny, const unsigned int nz)\n	\
{\n						\
  const int i = get_global_id(0);\n		\
  const int j = get_global_id(1);\n		\
  const int k = get_global_id(2);\n		\
  const int pos = i * nz * ny + j * nz + k;	\
  X[2 * pos] = i;				\
  X[2 * pos + 1] = j + k * k;			\
}\n";
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  set_kernel_arg(initkernel, 0, sizeof(cl_mem), &inbuf);
  set_kernel_arg(initkernel, 1, sizeof(unsigned int), &ny);
  set_kernel_arg(initkernel, 2, sizeof(unsigned int), &nz);

  std::cout << "Allocating " 
	    << fft.ncomplex() 
	    << " doubles." << std::endl;
  double *X = new double[2 * fft.ncomplex()];
  double *FX = new double[2 * fft.ncomplex()];

  cl_event clv_init = clCreateUserEvent(ctx, NULL);
  cl_event clv_toram = clCreateUserEvent(ctx, NULL);
  cl_event clv_forward = clCreateUserEvent(ctx, NULL);
  cl_event clv_backward = clCreateUserEvent(ctx, NULL);
  if(N == 0) { // Transform forwards and back, outputting the buffer.
    init(X, nx, ny, nz);
    
    std::cout << "\nInput:" << std::endl;
    if(nx * ny * nz <= maxout) 
      show3C(X, nx, ny, nz);
    else 
      std::cout << X[0] << std::endl;

    //fft.ram_to_cbuf(X, &inbuf, 0, NULL, &clv_init);
    size_t global_wsize[] = {nx, ny, nz};
    clEnqueueNDRangeKernel(queue,
			   initkernel,
			   3, // cl_uint work_dim,
			   NULL, // global_work_offset,
			   global_wsize, // global_work_size, 
			   NULL, // size_t *local_work_size, 
			   0, NULL, &clv_init);

    fft.forward(&inbuf, inplace ? NULL : &outbuf, 1, &clv_init, &clv_forward);
    fft.cbuf_to_ram(FX, inplace ? &inbuf :&outbuf, 1, &clv_forward, &clv_toram);
    clWaitForEvents(1, &clv_toram);
    
    std::cout << "\nTransformed:" << std::endl;
    if(nx * ny * nz <= maxout) 
      show3C(FX, nx, ny, nz);
    else 
      std::cout << X[0] << std::endl;

    fft.backward(inplace ? &inbuf : &outbuf,
		 inplace ? NULL : &outbuf, 
		 1, &clv_forward, &clv_backward);
    fft.cbuf_to_ram(X, &inbuf, 1, &clv_backward, &clv_toram);
    clWaitForEvents(1, &clv_toram);

    std::cout << "\nTransformed back:" << std::endl;
    if(nx * ny * nz <= maxout) 
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
    	double rdiff = FX[2 * i] - df[2 * i];
    	double idiff = FX[2 * i + 1] - df[2 * i + 1];
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
    double *T = new double[N];
  
    cl_ulong time_start, time_end;
    for(unsigned int i = 0; i < N; i++) {
      //init(X, nx, ny, nz);
      //fft.ram_to_cbuf(X, &inbuf, 0, NULL, &clv_init);
      size_t global_wsize[] = {nx, ny, nz};
      clEnqueueNDRangeKernel(queue,
			     initkernel,
			     3, // cl_uint work_dim,
			     NULL, // global_work_offset,
			     global_wsize, // global_work_size, 
			     NULL, // size_t *local_work_size, 
			     0, NULL, &clv_init);

      fft.forward(&inbuf, inplace ? NULL : &outbuf, 
		  1, &clv_init, &clv_forward);
      clWaitForEvents(1, &clv_forward);

      clGetEventProfilingInfo(clv_forward,
    			      CL_PROFILING_COMMAND_START,
    			      sizeof(time_start),
    			      &time_start, NULL);
      clGetEventProfilingInfo(clv_forward,
    			      CL_PROFILING_COMMAND_END,
    			      sizeof(time_end), 
    			      &time_end, NULL);
      T[i] = 1e-6 * (time_end - time_start);
    }
    timings("fft timing", nx, T, N,stats);
    delete[] T;
  }

  delete X;

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
