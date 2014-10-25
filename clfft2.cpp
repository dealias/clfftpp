#include <stdlib.h>

#include <iostream>
#include <timing.h>
#include <seconds.h>
#include <platform.hpp>
#include <clfft.hpp>

#include<vector>

#include <getopt.h>
#include <utils.h>

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
      X[2*pos + 1] = j;
    }
  }
}

int main(int argc, char* argv[]) {

  show_devices();

  int platnum=0;
  int devnum=0;

  bool time_copy=false;

  int nx = 4;
  int ny = 4;
  //nx=262144;

  int N=10;

  unsigned int stats=0; // Type of statistics used in timing test.

#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"p:d:c:m:x:y:N:S:h");
    if (c == -1) break;
    
    switch (c) {
    case 'p':
      platnum=atoi(optarg);
      break;
    case 'd':
      devnum=atoi(optarg);
      break;
    case 'c':
      if(atoi(optarg) == 0)
	time_copy = false;
      else
	time_copy = true;
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
      usage(1);
      exit(0);
      break;
    default:
      std::cout << "Invalid option" << std::endl;
      usage(1);
      exit(1);
    }
  }


  std::vector<std::vector<cl_device_id> > dev_ids;
  create_device_tree(dev_ids);
  cl_device_id device = dev_ids[platnum][devnum];

  std::vector<cl_platform_id> plat_ids;
  find_platform_ids(plat_ids);
  cl_platform_id platform = plat_ids[devnum];

  cl_context ctx = create_context(platform, device);
  cl_command_queue queue = create_queue(ctx, device);
  
  clfft2 fft(nx,ny,queue,ctx);
  fft.create_clbuf();

  double *X = fft.create_rambuf();

  std::cout << "\nInput:" << std::endl;
  init(X,nx,ny);
  if(nx * ny <= 100) 
    show(X,nx,ny);
  else 
    std::cout << X[0] << std::endl;

  fft.ram_to_cl(X);
  fft.forward();
  fft.cl_to_ram(X);
  std::cout << "\nTransformed:" << std::endl;
  if(nx * ny <= 100) 
    show(X,nx,ny);
  else 
    std::cout << X[0] << std::endl;

  fft.ram_to_cl(X);
  fft.backward();
  fft.cl_to_ram(X);
  std::cout << "\nTransformed back:" << std::endl;
  if(nx * ny <= 100) 
    show(X,nx,ny);
  else 
    std::cout << X[0] << std::endl;


  std::cout << "\nTimings:" << std::endl;
  double *T=new double[N];
  if(time_copy) {
    for(int i=0; i < N; ++i) {
      init(X,nx,ny);
      seconds();
      fft.ram_to_cl(X);
      fft.forward();
      fft.wait();
      fft.cl_to_ram(X);
      T[i]=seconds();
    }
    timings("fft with copy",nx,T,N,stats);
  } else {
    for(int i=0; i < N; ++i) {
      init(X,nx,ny);
      seconds();
      fft.ram_to_cl(X);
      fft.forward();
      fft.wait();
      fft.cl_to_ram(X);
      T[i]=seconds();
    }
    timings("fft without copy",nx,T,N,stats);
  }
  free(X);

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
