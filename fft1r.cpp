#include <stdlib.h>
#include <platform.hpp>
#include <clfft.hpp>

#include <iostream>
#include <timing.h>
#include <seconds.h>

#include<vector>

#include <getopt.h>
#include <utils.h>

template<class T>
void showC(T *X, int n)
{
  unsigned int nc = n / 2 + 1;
  for(unsigned int i = 0; i < nc; ++i)
    std::cout << "(" << X[2 * i] << "," 
	      <<  X[2 * i + 1] << ")" 
	      << std::endl;
}

template<class T>
void showR(const T *X, int n)
{
  for(unsigned int i = 0; i < n; ++i)
    std::cout << X[i] << std::endl;
}

template<class T>
void initR(T *X, int n)
{
  for(unsigned int i = 0; i < n; ++i)
    X[i] = i;
}

int main(int argc, char *argv[]) {
  show_devices();

  int platnum = 0;
  int devnum = 0;
  bool time_copy = false;
  int nx = 4;
  int N = 0;
  unsigned int stats = 0; // Type of statistics used in timing test.

  int maxout = 32; // maximum size of array output in entirety

#ifdef __GNUC__	
  optind = 0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"p:d:m:x:N:S:h");
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
  cl_platform_id platform = plat_ids[platnum];

  cl_context ctx = create_context(platform, device);
  cl_command_queue queue = create_queue(ctx, device, CL_QUEUE_PROFILING_ENABLE);

  clfft1r fft(nx, queue, ctx);

  std::cout << "Allocating " 
	    << fft.get_ncomplexfloats() 
	    << " doubles for complex." << std::endl;
  double *Xin = new double[fft.get_ncomplexfloats()];
  std::cout << "Allocating " 
	    << fft.get_nrealfloats() 
	    << " doubles for real." << std::endl;
  double *Xout = new double[fft.get_nrealfloats()];

  cl_event r2c_event = clCreateUserEvent(ctx, NULL);
  cl_event c2r_event = clCreateUserEvent(ctx, NULL);
  cl_event forward_event = clCreateUserEvent(ctx, NULL);
  cl_event backward_event = clCreateUserEvent(ctx, NULL);

  if(N == 0) {
    std::cout << "\nInput:" << std::endl;
    initR(Xin, nx);
    if(nx <= maxout)
      showR(Xin, nx);
    else
      std::cout << Xin[0] << std::endl;

    fft.ram_to_inbuf(Xin, &r2c_event);
    fft.forward(1, &r2c_event, &forward_event);
    fft.outbuf_to_ram(Xout, 1, &forward_event, &c2r_event);
    clWaitForEvents(1, &c2r_event);

    std::cout << "\nTransformed:" << std::endl;
    if(nx <= maxout)
      showC(Xout, nx);
    else 
      std::cout << Xout[0] << std::endl;

    fft.backward(1, &forward_event, &backward_event);
    fft.inbuf_to_ram(Xin, 1, &backward_event, &c2r_event);
    clWaitForEvents(1, &c2r_event);

    std::cout << "\nTransformed back:" << std::endl;
    if(nx <= maxout) 
      showR(Xin, nx);
    else 
      std::cout << Xin[0] << std::endl;
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
  
