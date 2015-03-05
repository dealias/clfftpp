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
  unsigned int nc=n/2+1;
  for(unsigned int i=0; i < nc; ++i) {
    std::cout << "(" << X[2*i] << "," <<  X[2*i +1] << ")" << std::endl;
  }
}

template<class T>
void showR(T *X, int n)
{
  for(unsigned int i=0; i < n; ++i) {
    std::cout <<  X[i] << std::endl;
  }
}

template<class T>
void initR(T *X, int n)
{
  for(unsigned int i=0; i < n; ++i) {
    X[i] = i;
  }
}

int main(int argc, char* argv[]) {
  show_devices();

  int platnum=0;
  int devnum=0;
  bool time_copy=false;
  int nx = 4;
  int N=10;
  unsigned int stats=0; // Type of statistics used in timing test.

#ifdef __GNUC__	
  optind=0;
#endif	
  for (;;) {
    int c = getopt(argc,argv,"p:d:m:x:N:S:h");
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
    case 'm':
      nx=atoi(optarg);
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
  cl_platform_id platform = plat_ids[platnum];

  cl_context ctx = create_context(platform, device);
  cl_command_queue queue = create_queue(ctx, device, CL_QUEUE_PROFILING_ENABLE);

  clfft1r fft(nx, queue, ctx);
  fft.create_clbuf();

  //  typedef float real;

  double *X = new double[fft.get_nfloats()];

  cl_event r2c_event = clCreateUserEvent(ctx, NULL);
  cl_event c2r_event = clCreateUserEvent(ctx, NULL);
  cl_event forward_event = clCreateUserEvent(ctx, NULL);
  cl_event backward_event = clCreateUserEvent(ctx, NULL);

  std::cout << "\nInput:" << std::endl;
  initR(X,nx);
  if(nx <= 32) 
    showR(X,nx);
  else 
    std::cout << X[0] << std::endl;

  fft.ram_to_cl(X, &r2c_event);
  
  // cl_int ret = clSetUserEventStatus(forward_event,  CL_COMPLETE);
  // if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
  // assert(ret == CL_SUCCESS);
  fft.forward(1, &r2c_event, &forward_event);

  fft.cl_to_ram(X, 1, &forward_event, &c2r_event);
  clWaitForEvents(1, &c2r_event);
  std::cout << "\nTransformed:" << std::endl;
  if(nx <= 32)
    showC(X,nx);
  else 
    std::cout << X[0] << std::endl;
  
  fft.backward(1, &forward_event, &backward_event);
  fft.cl_to_ram(X, 1, &backward_event, &c2r_event);
  clWaitForEvents(1, &c2r_event);
  std::cout << "\nTransformed back:" << std::endl;
  if(nx <= 32) 
    showR(X,nx);
  else 
    std::cout << X[0] << std::endl;

  if(N > 0) {
    std::cout << "\nTimings:" << std::endl;
    double *T=new double[N];

    cl_ulong time_start, time_end;
    for(int i=0; i < N; ++i) {
      initR(X,nx);
      seconds();
      fft.ram_to_cl(X, &r2c_event);
      fft.forward(1, &r2c_event, &forward_event);
      fft.cl_to_ram(X, 1, &forward_event, &c2r_event);
      clWaitForEvents(1, &c2r_event);

      if(time_copy) {
	clGetEventProfilingInfo(r2c_event,
				CL_PROFILING_COMMAND_START,
				sizeof(time_start),
				&time_start, NULL);
	clGetEventProfilingInfo(c2r_event,
				CL_PROFILING_COMMAND_END,
				sizeof(time_end), 
				&time_end, NULL);
      } else {
	clGetEventProfilingInfo(forward_event,
				CL_PROFILING_COMMAND_START,
				sizeof(time_start),
				&time_start, NULL);
	clGetEventProfilingInfo(forward_event,
				CL_PROFILING_COMMAND_END,
				sizeof(time_end), 
				&time_end, NULL);
      }
      T[i] = 1e-6 * (time_end - time_start);
    }
    if(time_copy) 
      timings("fft with copy",nx,T,N,stats);
    else 
      timings("fft without copy",nx,T,N,stats);
    delete[] T;
  }
  free(X);

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
