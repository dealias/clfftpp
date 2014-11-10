#include <string>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <platform.hpp>

#include <clutils.h>
#include <timing.h>
#include <seconds.h>
#include <assert.h>

#include <fft.hpp>

#include <getopt.h>
#include <CL/cl.hpp>

// For fftw++ comparison
#include "Complex.h"
#include "fftw++.h"
#include "Array.h"

inline void init(Array::array2<Complex>& f, unsigned int mx, unsigned int my) 
{
  for(unsigned int i=0; i < mx; ++i)
    for(unsigned int j=0; j < my; j++)
      f[i][j]=Complex(i,j);
}

void read_file(std::string &str, const char* filename)
{
  std::ifstream t(filename);

  t.seekg(0, std::ios::end);
  str.reserve(t.tellg());
  t.seekg(0, std::ios::beg);
  
  str.assign(std::istreambuf_iterator<char>(t),
	     std::istreambuf_iterator<char>());
}

void check_cl_ret(cl_int ret, const char* msg)
{
  if(ret != CL_SUCCESS) {
    
    std::cerr << "ERROR: " << msg 
	      << " with retval: " << ret 
	      << " : " << clErrorString(ret) 
	      << std::endl;
  }
  assert(ret == CL_SUCCESS);
}

template< class T>
void init(const unsigned int nx, const unsigned int ny, T *f)
{
  for(unsigned int ix=0; ix < nx; ++ix) {
    unsigned int iy;
    for(iy=0; iy < ny; ++iy) {
      unsigned int pos=2*(ix*ny + iy);
      f[pos]=ix;
      f[pos+1]=iy; /* iy+1; */
    }
  }
}

template< class T>
void show(const unsigned int nx, const unsigned int ny, T *f, 
	  unsigned int outlimit)
{

  if(nx*ny < outlimit) {
    for(unsigned int ix=0; ix < nx; ++ix) {
      for(unsigned int iy=0; iy < ny; ++iy) {
	int pos=2*(ix*ny + iy); 
	std::cout << "(" << f[pos] << "," << f[pos+1] << ") ";
      }
      std::cout << std::endl;
    }
  } else {
    std::cout << "(" << f[0] << "," << f[1] << ") " << std::endl;
  }
}


int main(int argc, char* argv[])
{
  // Set up the OpenCL device, platform, queue, and context.
  show_devices();

  int platnum=0;
  int devnum=0;
  
  unsigned int nx = 4;
  unsigned int ny = 4;
  //nx=262144;

  unsigned int N=10;

  unsigned int stats=MEAN; // Type of statistics used in timing test.

  for (;;) {
    int c = getopt(argc,argv,"p:d:m:x:y:N:S:h");
    if (c == -1) break;
    
    switch (c) {
    case 'p':
      platnum=atoi(optarg);
      break;
    case 'd':
      devnum=atoi(optarg);
      break;
    case 'x':
      nx=atoi(optarg);
      break;
    case 'y':
      ny=atoi(optarg);
      break;
    case 'm':
      nx=atoi(optarg);
      ny=atoi(optarg);
      break;
    case 'N':
      N=atoi(optarg);
      break;
    case 'S':
      nx=atoi(optarg);
      break;
    case 'h':
      //usage(1);
      exit(0);
      break;
    default:
      std::cout << "Invalid option" << std::endl;
      //usage(1);
      exit(1);
    }
  }

  std::vector<std::vector<cl_device_id> > dev_ids;
  create_device_tree(dev_ids);
  cl_device_id device = dev_ids[platnum][devnum];

  std::vector<cl_platform_id> plat_ids;
  find_platform_ids(plat_ids);
  cl_platform_id platform = plat_ids[devnum];

  const cl_context ctx = create_context(platform, {device});
  const cl_command_queue queue = create_queue(ctx, {device});

  unsigned int outlimit=100;

  // bool pdouble=false;
  // auto ff = pdouble ? (double) 1 : (float)1;
  // std::cout << sizeof(ff) << std::endl;

  typedef double REAL;
  //typedef float REAL;

  REAL *f=new REAL[2*nx*ny];

  //mfft1d <float>fft(platnum,devnum,nx,ny);
  mfft1d <REAL>fft(queue,ctx,device,nx,ny);
  fft.build();
  fft.alloc_rw();
  fft.set_args();

  std::cout << "Input:" << std::endl;
  init(nx,ny,f);
  show(nx,ny,f,outlimit);
  fft.write_buffer(f);
  fft.forward();
  fft.finish();
  fft.read_buffer(f);
  std::cout << "\nOutput:" << std::endl;
  show(nx,ny,f,outlimit);

  {
    std::cout << "\nOutput of mfft1d using FFTW++:" << std::endl; 

    size_t align=sizeof(Complex);
    Array::array2<Complex> F(nx,ny,align);
    fftwpp::mfft1d Forward(ny,-1,nx,1,ny);
    init(F,nx,ny);
    Forward.fft(F);
    if(nx*ny < outlimit) {
      for(unsigned int i=0; i < nx; i++) {
	for(unsigned int j=0; j < ny; j++)
	  std::cout << F[i][j] << "\t";
	std::cout << std::endl;
      }
    } else { 
      std::cout << F[0][0] << std::endl;
    }

    double err=0.0;
    for(unsigned int i=0; i < nx; i++) {
      for(unsigned int j=0; j < ny; j++) {
	double fr = F[i][j].re - f[2*(i*ny +j)];
	double fi = F[i][j].im - f[2*(i*ny +j)+1];
	err += fr*fr + fi*fi;
      }
    }
    err = sqrt(err/(double)(nx*ny));
    
    std::cout << "\nL2 difference: " << err << std::endl;
    
  }

  init(nx,ny,f);
  double *T=new double[N];
  for(unsigned int i=0; i < N; ++i) {
    // FIXME
    seconds();
    // FIXME
    T[i]=seconds();
  }
  //timings("mfft1d",nx,T,N,stats);

  

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);
}
