#include <stdlib.h>

#include <iostream>
#include <timing.h>
#include <seconds.h>
#include <platform.hpp>
#include <clfft.hpp>

#include<vector>

template<class T>
void show(T *X, int nx, int ny)
{
  for(unsigned int i=0; i < nx; ++i) {
    for(unsigned int j=0; j < ny; ++j) {
      unsigned pos = i*ny + j; 
      std::cout << "(" << X[2*pos] 
		<< "," <<  X[2*pos +1] << ") ";

    }
    std::cout << std::endl;
  }
}

template<class T>
void init(T *X, int nx, int ny)
{
  for(unsigned int i=0; i < nx; ++i) {
    for(unsigned int j=0; j < ny; ++j) {
      unsigned pos = i*ny + j; 
      X[2*pos] = i;
      X[2*pos + 1] = 0.0;
    }
  }
}

int main() {

  show_devices();

  int platnum=0;
  int devnum=0;

  std::vector<std::vector<cl_device_id> > dev_ids;
  create_device_tree(dev_ids);
  cl_device_id device = dev_ids[platnum][devnum];

  std::vector<cl_platform_id> plat_ids;
  find_platform_ids(plat_ids);
  cl_platform_id platform = plat_ids[devnum];

  cl_context ctx = create_context(platform, device);
  cl_command_queue queue = create_queue(ctx, device);

  int nx = 4;
  int ny = 4;
  //nx=262144;
  
  typedef float real;

  real *X = (real *)malloc(nx * ny * 2 * sizeof(real));

  init(X,nx,ny);
  //show(X,nx);

  clfft2 fft(nx,ny,queue,ctx);
  fft.create_clbuf();

  fft.ram_to_cl(X);
  fft.forward();
  fft.cl_to_ram(X);
  if(nx * ny  <= 100) {
    show(X,nx,ny);
  } else {
    std::cout << X[0] << std::endl;
  }

  int N=10;
  double *T=new double[N];

  for(int i=0; i < N; ++i) {
    init(X,nx,ny);
    seconds();
    fft.ram_to_cl(X);
    fft.forward();
    fft.wait();
    fft.cl_to_ram(X);
    T[i]=seconds();
  }
  timings("fft with copy",nx,T,N,MEDIAN);

  for(int i=0; i < N; ++i) {
    init(X,nx,ny);
    seconds();
    fft.ram_to_cl(X);
    fft.forward();
    fft.wait();
    fft.cl_to_ram(X);
    T[i]=seconds();
  }
  timings("fft without copy",nx,T,N,MEDIAN);

  free(X);

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
