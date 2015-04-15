#include <stdlib.h>
#include <platform.hpp>
#include <clfft.hpp>
#include <iostream>
#include "utils.hpp"

template<class T>
void init2R(T *X, unsigned int nx, unsigned int ny)
{
  for(unsigned int i = 0; i < nx; ++i) {
    for(unsigned int j = 0; j < ny; ++j) {
      unsigned int pos = i * ny + j; 
      X[pos] = i * i + j;
    }
  }
}

int main() {
  int platnum = 0;
  int devnum = 0;
  unsigned int nx = 4;
  unsigned int ny = 4;
  bool inplace = false;

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

  clfft2r fft(nx, ny, inplace, queue, ctx);
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

  std::cout << "\nInput:" << std::endl;
  init2R(Xin, nx, ny);
  show2R(Xin, nx, ny);
    
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
  show2H(Xout, fft.ncomplex(0), fft.ncomplex(1), inplace ? 1 : 0);

  if(inplace) {
    fft.backward(&inbuf, NULL, 1, &forward_event, &backward_event);
    fft.cbuf_to_ram(Xin, &inbuf, 1, &backward_event, &c2r_event);
  } else {
    fft.backward(&outbuf, &inbuf, 1, &forward_event, &backward_event);
    fft.rbuf_to_ram(Xin, &inbuf, 1, &backward_event, &c2r_event);
  }
  clWaitForEvents(1, &c2r_event);

  std::cout << "\nTransformed back:" << std::endl;
  show2R(Xin, nx, ny);
  delete[] Xout;
  delete[] Xin;
  
  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
