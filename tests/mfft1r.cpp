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
    n = nx;
    M = ny;
    istride = nx;
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
    idist = ny;
    odist = ny / 2 + 1;
    break;
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

  if(istride == 0) istride = 1;
  if(ostride == 0) ostride = 1;
  if(idist == 0) idist = nx;
  if(odist == 0) odist = nx / 2 + 1;
  if(n == 0) n = nx;
  if(M == 0) M = ny;
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
  
  clmfft1r fft(n, M, istride, ostride, idist, odist, inplace, 
	       queue, ctx);

  cout << endl;
  cl_mem inbuf, outbuf;
  fft.create_cbuf(&inbuf);
  if(inplace) {
    cout << "in-place transform" << endl;
  } else {
    cout << "out-of-place transform" << endl;
    fft.create_cbuf(&outbuf);
  }

   string init_source ="\
#pragma OPENCL EXTENSION cl_khr_fp64: enable\n			\
__kernel void init(__global double *X, const unsigned int nx)	\
{								\
  const int i = get_global_id(0);				\
  const int j = get_global_id(1);				\
  int pos = j * nx + i;						\
  X[pos] = i + 10 * j;						\
}";
  cl_program initprog = create_program(init_source, ctx);
  build_program(initprog, device);
  cl_kernel initkernel = create_kernel(initprog, "init"); 
  set_kernel_arg(initkernel, 0, sizeof(cl_mem), &inbuf);
  set_kernel_arg(initkernel, 1, sizeof(unsigned int), &nx);
  size_t global_wsize[] = {nx, ny}; 

  cout << "nx: " << nx << endl;
  cout << "M: " << M << endl;
  cout << "istride: " << istride << endl;
  cout << "ostride: " << ostride << endl;
  cout << "idist: " << idist << endl;
  cout << "odist: " << odist << endl;

  cout << "\nAllocating " 
  	    << nx * ny
  	    << " doubles for real." << endl;
  double *X = new double[nx * ny];  
  int np = (n / 2 + 1);
  int ncomplex = 2 * np * M;
  cout << "Allocating "
  	    << ncomplex
  	    << " doubles for complex." << endl;
  double *FX = new double[ncomplex];

  cl_event clv_init = clCreateUserEvent(ctx, NULL);
  cl_event clv_toram = clCreateUserEvent(ctx, NULL);
  cl_event clv_forward = clCreateUserEvent(ctx, NULL);
  cl_event clv_backward = clCreateUserEvent(ctx, NULL);
  if(N == 0) {
    tolerance *= log((double)nx + 1);
    cout << "Tolerance: " << tolerance << endl;

    cout << "\nInput:" << endl;
    //initR(X, nx, M);
    //fft.ram_to_rbuf(X, &inbuf, 0, NULL, &clv_init);

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
      cout << X[0] << endl;

    fft.forward(&inbuf, inplace ? NULL : &outbuf, 1, &clv_init, &clv_forward);
    fft.cbuf_to_ram(FX, inplace ? &inbuf : &outbuf, 
		    1, &clv_forward, &clv_toram);
    clWaitForEvents(1, &clv_toram);

    cout << "\nTransformed:" << endl;
    if(nx <= maxout) {
      if(istride == 1 && ostride == 1) {
	show2C(FX, M, np);
      } else if (istride == nx && ostride == ny) {
	show2C(FX, np, M);
      } else {
	show2C(FX, np * M, 1);
      }
    } else {
      cout << FX[0] << endl;
    }    

    fft.backward(inplace ? &inbuf : & outbuf, 
		 inplace ? NULL : &inbuf, 1, &clv_forward, &clv_backward);
    fft.rbuf_to_ram(X, &inbuf, 1, &clv_backward, &clv_toram);
    clWaitForEvents(1, &clv_toram);

    cout << "\nTransformed back:" << endl;
    if(nx <= maxout) {
      show1R(X, n, M);
    } else {
      cout << X[0] << endl;
    }

    // Compute the round-trip error.
    {
      double *X0 = new double[fft.nreal()];
      clEnqueueNDRangeKernel(queue,
			     initkernel,
			     2, // cl_uint work_dim,
			     NULL, // global_work_offset,
			     global_wsize, // global_work_size, 
			     NULL, // size_t *local_work_size, 
			     0, NULL, &clv_init);
      fft.rbuf_to_ram(X0, &inbuf, 1, &clv_init, &clv_toram);
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
      // fftw::maxthreads=get_max_threads();
      size_t align = sizeof(Complex);
      Array::array2<double> f(nx, ny, align);
      Array::array2<Complex> g(nx, ny / 2 + 1, align);
      fftwpp::mrcfft1d Forward(n, M, istride, ostride, idist, odist, f, g);
      double *df = (double *)f();
      double *dg = (double *)g();
      clEnqueueNDRangeKernel(queue,
			     initkernel,
			     2, // cl_uint work_dim,
			     NULL, // global_work_offset,
			     global_wsize, // global_work_size, 
			     NULL, // size_t *local_work_size, 
			     0, NULL, &clv_init);
      fft.rbuf_to_ram(df, &inbuf, 1, &clv_init, &clv_toram);
      clWaitForEvents(1, &clv_toram);

      cout << endl;
      cout << "fftw++ input:" << endl;
      cout << f << endl;

      Forward.fft(f, g);

      cout << "fftw++ transformed:" << endl;
      if(istride == 1 && ostride == 1) {
	show2C(dg, M, np);
      } else if (istride == nx && ostride == ny) {
	show2C(dg, np, M);
      } else {
	show2C(dg, np * M, 1);
      }
      //cout << g << endl;

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

