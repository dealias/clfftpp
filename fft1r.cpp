#include <stdlib.h>
#include <platform.hpp>
#include <clfft.hpp>

#include <iostream>
#include <timing.h>
#include <seconds.h>

#include<vector>

#include <getopt.h>
#include "utils.hpp"

template<class T>
void initR(T *X, int n)
{
  for(unsigned int i = 0; i < n; ++i)
    X[i] = i;
}

int main(int argc, char *argv[]) {

  int platnum = 0;
  int devnum = 0;
  //bool time_copy = false;
  int nx = 4;
  int N = 0;
  unsigned int stats = 0; // Type of statistics used in timing test.
  bool inplace = false;

  int maxout = 32; // maximum size of array output in entirety

#ifdef __GNUC__	
  optind = 0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"p:d:m:x:N:S:hi:");
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
  if(inplace)
    std::cout << "in-place transform" << std::endl;
  else
    std::cout << "out-of-place transform" << std::endl;
  std::cout << "Allocating " 
	    << fft.nreal() 
	    << " doubles for real." << std::endl;
  double *Xin = new double[fft.nreal()];
  std::cout << "Allocating " 
	    << 2 * fft.ncomplex()
	    << " doubles for complex." << std::endl;
  double *Xout = new double[2 * fft.ncomplex()];
  fft.create_inbuf();
  if(!inplace)
    fft.create_outbuf();

  cl_event r2c_event = clCreateUserEvent(ctx, NULL);
  cl_event c2r_event = clCreateUserEvent(ctx, NULL);
  cl_event forward_event = clCreateUserEvent(ctx, NULL);
  cl_event backward_event = clCreateUserEvent(ctx, NULL);

  if(N == 0) {
    std::cout << "\nInput:" << std::endl;
    initR(Xin, nx);
    if(nx <= maxout)
      show1R(Xin, nx);
    else
      std::cout << Xin[0] << std::endl;

    fft.ram_to_input(Xin, &r2c_event);
    fft.forward(1, &r2c_event, &forward_event);
    fft.output_to_ram(Xout, 1, &forward_event, &c2r_event);
    clWaitForEvents(1, &c2r_event);

    std::cout << "\nTransformed:" << std::endl;
    if(nx <= maxout)
      show1C(Xout, fft.ncomplex(0));
    else 
      std::cout << Xout[0] << std::endl;
    
    fft.finish();
    fft.backward(&backward_event);
    fft.input_to_ram(Xin, 1, &backward_event, &c2r_event);
    clWaitForEvents(1, &c2r_event);
    
    std::cout << "\nTransformed back:" << std::endl;
    if(nx <= maxout) 
      show1R(Xin, nx);
    else 
      std::cout << Xin[0] << std::endl;

    fft.output_to_ram(Xout);
    fft.finish();
    show1C(Xout, fft.ncomplex(0));

    std::cout << "done" << std::endl;

  } else {
    // FIXME: put timing stuff here.
  }
  delete[] Xin;
  delete[] Xout;

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
