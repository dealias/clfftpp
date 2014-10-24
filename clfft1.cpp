#include <stdlib.h>

#include <iostream>
#include <timing.h>
#include <seconds.h>
#include <platform.hpp>
#include <clfft.hpp>


#include<vector>




template<class T>
void show(float *X, int n)
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
  std::cout << std::endl;

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

  typedef float real;

  int buf_size= nx * 2 * sizeof(real);
  real *X = (real *)malloc(buf_size);

  int N=10;
  double *T=new double[N];

  init(X,nx);
  //show(X,nx);

  cl_int err;
  /* Prepare OpenCL memory objects and place data inside them. */
  cl_mem bufX = clCreateBuffer(ctx, 
			       CL_MEM_READ_WRITE, 
			       buf_size,
			       NULL,
			       &err);

  clfft1 fft1(nx,queue,ctx,bufX);
  // Copy X to bufX
  err = clEnqueueWriteBuffer(queue,
			     bufX,
			     CL_TRUE,
			     0,
			     buf_size,
			     X,
			     0,
			     NULL,
			     NULL);

  for(int i=0; i < N; ++i) {
    init(X,nx);

    seconds();

    // Copy X to bufX
    err = clEnqueueWriteBuffer(queue,
			       bufX,
			       CL_TRUE,
			       0,
			       buf_size,
			       X,
			       0,
			       NULL,
			       NULL);

    // Execute the plan.
    fft1.fft();

    // Wait for calculations to be finished.
    err = clFinish(queue);
    
    // Fetch results of calculations.
    err = clEnqueueReadBuffer(queue, 
			      bufX, 
			      CL_TRUE, 
			      0, 
			      buf_size,
			      X, 
			      0, 
			      NULL, 
			      NULL );

    T[i]=seconds();
  }
  timings("with copy",nx,T,N,MEDIAN);

  for(int i=0; i < N; ++i) {
    init(X,nx);

    // Copy X to bufX
    err = clEnqueueWriteBuffer(queue,
			       bufX,
			       CL_TRUE,
			       0,
			       buf_size,
			       X,
			       0,
			       NULL,
			       NULL);
    seconds();

    // Execute the plan.
    fft1.fft();

    // Wait for calculations to be finished.
    err = clFinish(queue);

    T[i]=seconds();

    // Fetch results of calculations.
    err = clEnqueueReadBuffer(queue, 
			      bufX, 
			      CL_TRUE, 
			      0, 
			      buf_size,
			      X, 
			      0, 
			      NULL, 
			      NULL );

  }

  timings("without copy",nx,T,N,MEDIAN);

  /* Release OpenCL memory objects. */
  clReleaseMemObject(bufX);

  free(X);

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
