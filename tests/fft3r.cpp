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

template<class T>
void init3R(T *X, unsigned int nx, unsigned int ny, unsigned int nz)
{
  for(unsigned int i = 0; i < nx; ++i) {
    for(unsigned int j = 0; j < ny; ++j) {
      for(unsigned int k = 0; k < nz; ++k) {
	unsigned int pos = i * ny * nz + j * nz + k;
	X[pos] = i * i + j + 10 * k;
      }
    }
  }
}

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
  
  unsigned int maxout = 32; // maximum size of array output in entirety

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
  cl_command_queue queue = create_queue(ctx, device, CL_QUEUE_PROFILING_ENABLE);

  clfft3r fft(nx, ny, nz, inplace, queue, ctx);
  cl_mem inbuf, outbuf;
  if(inplace) {
    fft.create_cbuf(&inbuf);
    cout << "in-place transform" << endl;
  } else {
    cout << "out-of-place transform" << endl;
    fft.create_rbuf(&inbuf);
    fft.create_cbuf(&outbuf);
  }

  string init_source ="\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n	\
__kernel void init(__global double *X, \
const unsigned int ny, const unsigned int nz)	\
{						\
  const int i = get_global_id(0);		\
  const int j = get_global_id(1);		\
  const int k = get_global_id(2);		\
  unsigned int pos = i * ny * nz + j * nz + k;	\
  X[pos] = i * i + j + 10 * k;			\
}";
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  set_kernel_arg(initkernel, 0, sizeof(cl_mem), &inbuf);
  set_kernel_arg(initkernel, 1, sizeof(unsigned int), &ny);
  set_kernel_arg(initkernel, 2, sizeof(unsigned int), &nz);

  cout << "Allocating " 
	    << (inplace ? 2 * fft.ncomplex() : fft.nreal())
	    << " doubles for real." << endl;
  double *X = new double[inplace ? 2 * fft.ncomplex() : fft.nreal()];
  cout << "Allocating "
	    << 2 * fft.ncomplex() 
	    << " doubles for complex." << endl;
  double *FX = new double[2 * fft.ncomplex()];

  // Create OpenCL events
  cl_event clv_init = clCreateUserEvent(ctx, NULL);
  cl_event clv_toram = clCreateUserEvent(ctx, NULL);
  cl_event clv_forward = clCreateUserEvent(ctx, NULL);
  cl_event clv_backward = clCreateUserEvent(ctx, NULL);

  if(N == 0) {
    tolerance *= log((double) max(max(nx, ny), nz) + 1);
    cout << "Tolerance: " << tolerance << endl;

    cout << "\nInput:" << endl;
    init3R(X, nx, ny, nz);
    if(nx * ny * nz <= maxout)
      show3R(X, nx, ny, nz);
    else
      cout << X[0] << endl;
    
    //fft.ram_to_rbuf(X, &inbuf, 0, NULL, &clv_init);
    size_t global_wsize[] = {nx, ny, nz};
    clEnqueueNDRangeKernel(queue,
			   initkernel,
			   3, // cl_uint work_dim,
			   NULL, // global_work_offset,
			   global_wsize, // global_work_size, 
			   NULL, // size_t *local_work_size, 
			   0, NULL, &clv_init);

    fft.forward(&inbuf, inplace ? NULL : &outbuf, 1, &clv_init, &clv_forward);
    fft.cbuf_to_ram(FX, inplace? &inbuf : &outbuf, 1, &clv_forward, &clv_toram);
    clWaitForEvents(1, &clv_toram);
    
    cout << "\nTransformed:" << endl;
    if(nx * ny * nz <= maxout) {
      show3H(FX, fft.ncomplex(0), fft.ncomplex(1), fft.ncomplex(2), 
	     inplace ? 1 : 0);
    } else {
      cout << FX[0] << endl;
    }

    fft.backward(inplace ? &inbuf : &outbuf, 
		 inplace? NULL : &inbuf, 1, &clv_forward, &clv_backward);
    if(inplace)
      fft.cbuf_to_ram(X, &inbuf, 1, &clv_backward, &clv_toram);
    else
      fft.rbuf_to_ram(X, &inbuf, 1, &clv_backward, &clv_toram);
    clWaitForEvents(1, &clv_toram);

    cout << "\nTransformed back:" << endl;
    // if(nx <= maxout)
    //   show3R(X, nx, ny, nz);
    // else 
    //   cout << X[0] << endl;

    // compute the round-trip error.
    {
      double *X0 = new double[inplace ? 2 * fft.ncomplex() : fft.nreal()];
      init3R(X0, nx, ny, nz);
      //show3R(X0, nx, ny, nz);
      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < fft.nreal(); ++i) {
    	double diff = fabs(X[i] - X0[i]);
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
      fftwpp::fftw::maxthreads = get_max_threads();
      size_t align = sizeof(Complex);
      unsigned int nzp = nz / 2 + 1;
      Array::array3<double> f(nx, ny, nz, align);
      Array::array3<Complex> g(nx, ny, nzp, align);
      fftwpp::rcfft3d Forward(nx, ny, nz, f, g);
      fftwpp::crfft3d Backward(nx, ny, nz, g, f);
    
      double *df = (double *)f();
      double *dg = (double *)g();
      init3R(df, nx, ny, nz);
      //show1C(df, nx);
      Forward.fft(f, g);
      //show1C(df, nx);
      
      //cout << "g:\n" << g << endl;
      
      unsigned int nzpskip = nzp + inplace;
      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < nx; ++i) {
      	for(unsigned int j = 0; j < ny; ++j) {
	  for(unsigned int k = 0; k < nzp; ++k) {
	    int pos = i * ny * nzp + j * nzp + k;
	    int pos0 = i * ny * nzpskip + j * nzpskip + k;
	    // cout << "pos0:  " << pos0  
	    // 	      << "\t(" << FX[2 * pos0] 
	    // 	      << " " << FX[2 * pos0 + 1]
	    // 	      << ")" << endl;
	    // cout << "pos:   " << pos  
	    // 	      << "\t(" << dg[2 * pos] 
	    // 	      << " " << dg[2 * pos + 1]
	    // 	      << ")" << endl;
	    // cout << endl;
	    double rdiff = FX[2 * pos0] - dg[2 * pos];
	    double idiff = FX[2 * pos0 + 1] - dg[2 * pos + 1];
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

  } else {
    double *T = new double[N];
  
    cl_ulong time_start, time_end;
    for(unsigned int i = 0; i < N; i++) {
      //init3R(X, nx, ny, nz);
      //fft.ram_to_rbuf(X, &inbuf, 0, NULL, &clv_init);
      size_t global_wsize[] = {nx, ny, nz};
      clEnqueueNDRangeKernel(queue,
			     initkernel,
			     3, // cl_uint work_dim,
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
  delete[] FX;
  delete[] X;
  
  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return error;
}
  
