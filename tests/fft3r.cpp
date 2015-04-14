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
  bool time_copy = false;
  unsigned int nx = 4;
  unsigned int ny = 4;
  unsigned int nz = 4;
  unsigned int N = 0;
  unsigned int stats = 0; // Type of statistics used in timing test.
  bool inplace = false;

  unsigned int maxout = 32; // maximum size of array output in entirety

#ifdef __GNUC__	
  optind = 0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"p:d:m:x:y:z:N:S:hi:");
    if (c == -1) break;
    
    switch (c) {
    case 'p':
      platnum = atoi(optarg);
      break;
    case 'd':
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
      usage(2);
      exit(0);
      break;
    default:
      std::cout << "Invalid option" << std::endl;
      usage(2);
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

  clfft3r fft(nx, ny, nz, inplace, queue, ctx);
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
	    << (inplace ? 2 * fft.ncomplex() : fft.nreal())
	    << " doubles for real." << std::endl;
  double *Xin = new double[inplace ? 2 * fft.ncomplex() : fft.nreal()];
  std::cout << "Allocating "
	    << 2 * fft.ncomplex() 
	    << " doubles for complex." << std::endl;
  double *Xout = new double[2 * fft.ncomplex()];

  // Create OpenCL events
  cl_event r2c_event = clCreateUserEvent(ctx, NULL);
  cl_event c2r_event = clCreateUserEvent(ctx, NULL);
  cl_event forward_event = clCreateUserEvent(ctx, NULL);
  cl_event backward_event = clCreateUserEvent(ctx, NULL);

  if(N == 0) {
    std::cout << "\nInput:" << std::endl;
    init3R(Xin, nx, ny, nz);
    if(nx * ny * nz <= maxout)
      show3R(Xin, nx, ny, nz);
    else
      std::cout << Xin[0] << std::endl;
    
    fft.ram_to_rbuf(Xin, &inbuf, 0, NULL, &r2c_event);
    if(inplace) {
      fft.forward(&inbuf, NULL, 1, &r2c_event, &forward_event);
      clWaitForEvents(1, &forward_event);
      fft.cbuf_to_ram(Xout, &inbuf, 1, &forward_event, &c2r_event);
    } else {
      fft.forward(&inbuf, &outbuf, 1, &r2c_event, &forward_event);
      fft.cbuf_to_ram(Xout, &outbuf, 1, &forward_event, &c2r_event);
    }
    clWaitForEvents(1, &c2r_event);
    
    std::cout << "\nTransformed:" << std::endl;
    if(nx * ny * nz <= maxout) {
      show3H(Xout, fft.ncomplex(0), fft.ncomplex(1), fft.ncomplex(2), 
	     inplace ? 1 : 0);
    } else {
      std::cout << Xout[0] << std::endl;
    }

    if(inplace) {
      fft.backward(&inbuf, NULL, 1, &forward_event, &backward_event);
      fft.cbuf_to_ram(Xin, &inbuf, 1, &backward_event, &c2r_event);
    } else {
      fft.backward(&outbuf, &inbuf, 1, &forward_event, &backward_event);
      fft.rbuf_to_ram(Xin, &inbuf, 1, &backward_event, &c2r_event);
    }
    clWaitForEvents(1, &c2r_event);

    std::cout << "\nTransformed back:" << std::endl;
    // if(nx <= maxout)
    //   show3R(Xin, nx, ny, nz);
    // else 
    //   std::cout << Xin[0] << std::endl;

    // compute the round-trip error.
    {
      double *X0 = new double[inplace ? 2 * fft.ncomplex() : fft.nreal()];
      init3R(X0, nx, ny, nz);
      //show3R(X0, nx, ny, nz);
      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < fft.nreal(); ++i) {
    	double diff = fabs(Xin[i] - X0[i]);
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
      
      //std::cout << "g:\n" << g << std::endl;
      
      unsigned int nzpskip = nzp + inplace;
      double L2error = 0.0;
      double maxerror = 0.0;
      for(unsigned int i = 0; i < nx; ++i) {
      	for(unsigned int j = 0; j < ny; ++j) {
	  for(unsigned int k = 0; k < nzp; ++k) {
	    int pos = i * ny * nzp + j * nzp + k;
	    int pos0 = i * ny * nzpskip + j * nzpskip + k;
	    // std::cout << "pos0:  " << pos0  
	    // 	      << "\t(" << Xout[2 * pos0] 
	    // 	      << " " << Xout[2 * pos0 + 1]
	    // 	      << ")" << std::endl;
	    // std::cout << "pos:   " << pos  
	    // 	      << "\t(" << dg[2 * pos] 
	    // 	      << " " << dg[2 * pos + 1]
	    // 	      << ")" << std::endl;
	    // std::cout << std::endl;
	    double rdiff = Xout[2 * pos0] - dg[2 * pos];
	    double idiff = Xout[2 * pos0 + 1] - dg[2 * pos + 1];
	    double diff = sqrt(rdiff * rdiff + idiff * idiff);
	    L2error += diff * diff;
	    if(diff > maxerror)
	      maxerror = diff;
	  }
	  //std::cout << std::endl;
	}
      }
      L2error = sqrt(L2error / (double) (nx * ny * nzp));

      std::cout << std::endl;
      std::cout << "Error with respect to FFTW:"  << std::endl;
      std::cout << "L2 error: " << L2error << std::endl;
      std::cout << "max error: " << maxerror << std::endl;
    }

  } else {
    // FIXME: put timing stuff here.
  }
  delete[] Xout;
  delete[] Xin;
  
  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
