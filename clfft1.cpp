#include <stdlib.h>

#include <iostream>
#include <timing.h>
#include <seconds.h>
#include <platform.hpp>
#include <clfft.hpp>

#include<vector>

template<class T>
void show(T *X, int n)
{
  for(unsigned int i=0; i < n; ++i) {
    std::cout << "(" << X[2*i] << "," <<  X[2*i +1] << ")" << std::endl;
  }
}

template<class T>
void init(T *X, int n)
{
  for(unsigned int i=0; i < n; ++i) {
    X[2*i] = i;
    X[2*i + 1] = 0.0;
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

  int nx = 1024;
  //nx=262144;

  nx=4;

  typedef float real;

  real *X = (real *)malloc(nx * 2 * sizeof(real));

  init(X,nx);
  //show(X,nx);

  clfft1 fft1(nx,queue,ctx);
  fft1.create_clbuf();

  fft1.ram_to_cl(X);
  fft1.forward();
  fft1.cl_to_ram(X);
  if(nx <= 32) 
    show(X,nx);
  else 
    std::cout << X[0] << std::endl;

  int N=10;
  double *T=new double[N];

  for(int i=0; i < N; ++i) {
    init(X,nx);
    seconds();
    fft1.ram_to_cl(X);
    fft1.forward();
    fft1.wait();
    fft1.cl_to_ram(X);
    T[i]=seconds();
  }
  timings("fft with copy",nx,T,N,MEDIAN);

  for(int i=0; i < N; ++i) {
    init(X,nx);
    seconds();
    fft1.ram_to_cl(X);
    fft1.forward();
    fft1.wait();
    fft1.cl_to_ram(X);
    T[i]=seconds();
  }
  timings("fft without copy",nx,T,N,MEDIAN);

  free(X);

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
