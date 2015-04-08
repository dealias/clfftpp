#include <stdlib.h>
#include <platform.hpp>
#include <clfft.hpp>

#include <iostream>

#include "utils.hpp"

template<class T>
void init(T *X, unsigned int n)
{
  for(unsigned int i = 0; i < n; ++i) {
    X[2 * i] = i;
    X[2 * i + 1] = 0.0;
  }
}

int main() {
  int platnum = 0;
  int devnum = 0;
  bool inplace = true;
  unsigned int nx = 4;

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
  
  clfft1 fft(nx, inplace, queue, ctx);
  cl_mem inbuf, outbuf;
  fft.create_cbuf(&inbuf);
  if(inplace) {
    std::cout << "in-place transform" << std::endl;
  } else {
    std::cout << "out-of-place transform" << std::endl;
    fft.create_cbuf(&outbuf);
  }
  
  std::cout << "Allocating " 
	    << 2 * fft.ncomplex() 
	    << " doubles." << std::endl;
  double *X = new double[2 * fft.ncomplex()];

  std::cout << "\nInput:" << std::endl;
  init(X, nx);
  show1C(X, nx);
  
  cl_event r2c_event = clCreateUserEvent(ctx, NULL);
  cl_event c2r_event = clCreateUserEvent(ctx, NULL);
  cl_event forward_event = clCreateUserEvent(ctx, NULL);
  cl_event backward_event = clCreateUserEvent(ctx, NULL);

  fft.ram_to_cbuf(X, &inbuf, 0, NULL, &r2c_event);
  if(inplace) {
    fft.forward(&inbuf, NULL, 1, &r2c_event, &forward_event);
    fft.cbuf_to_ram(X, &inbuf, 1, &forward_event, &r2c_event);
  } else {
    fft.forward(&inbuf, &outbuf, 1, &r2c_event, &forward_event);
    fft.cbuf_to_ram(X, &outbuf, 1, &forward_event, &r2c_event);
  }
  clWaitForEvents(1, &r2c_event);

  std::cout << "\nTransformed:" << std::endl;
  show1C(X, nx);
    
  if(inplace) {
    fft.backward(&inbuf, NULL, 1, &forward_event, &backward_event);
  } else {
    fft.backward(&outbuf, &inbuf, 1, &forward_event, &backward_event);
  }
  fft.cbuf_to_ram(X, &inbuf, 1, &backward_event, &c2r_event);
  clWaitForEvents(1, &c2r_event);

  std::cout << "\nTransformed back:" << std::endl;
  show1C(X, nx);

  delete[] X;
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
