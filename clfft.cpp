#include <stdlib.h>

#include <iostream>
#include <timing.h>
#include <seconds.h>
#include <platform.hpp>
#include <clfft.hpp>

#include<vector>

/* No need to explicitely include the OpenCL headers */
#include <clFFT.h>

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

void clfft_setup()
{
  cl_int err;  
  clfftSetupData fftSetup;
  err = clfftInitSetupData(&fftSetup);
  err = clfftSetup(&fftSetup);
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

  clfft_setup();

  int n = 1024;
  //n=262144;

  typedef float real;

  int buf_size= n * 2 * sizeof(real);
  real *X = (real *)malloc(buf_size);

  int N=10;
  double *T=new double[N];

  init(X,n);
  //show(X,n);

  cl_int err;    
  /* Prepare OpenCL memory objects and place data inside them. */
  cl_mem bufX = clCreateBuffer(ctx, 
			       CL_MEM_READ_WRITE, 
			       buf_size,
			       NULL,
			       &err);

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
  
  // Create a default plan for a complex FFT.
  clfftDim dim = CLFFT_1D;
  size_t clLengths[1] = {n};
  clfftPlanHandle plan;
  err = clfftCreateDefaultPlan(&plan, 
			       ctx, 
			       dim, 
			       clLengths);

  /* Set plan parameters. */
  err = clfftSetPlanPrecision(plan, 
			      CLFFT_SINGLE);
  err = clfftSetLayout(plan, 
		       CLFFT_COMPLEX_INTERLEAVED, 
		       CLFFT_COMPLEX_INTERLEAVED);
  err = clfftSetResultLocation(plan, 
			       CLFFT_INPLACE);

  // Bake the plan.
  err = clfftBakePlan(plan,
		      1, // numQueues: number of experiments 
		      &queue, // commQueueFFT
		      NULL, // Always NULL
		      NULL // Always NULL
		      );

  for(int i=0; i < N; ++i) {
    init(X,n);

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
    err = clfftEnqueueTransform(plan,
				CLFFT_FORWARD,
				1,
				&queue,
				0,
				NULL,
				NULL,
				&bufX,
				NULL,
				NULL);

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
  timings("with copy",n,T,N,MEDIAN);

  for(int i=0; i < N; ++i) {
    init(X,n);

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
    err = clfftEnqueueTransform(plan,
				CLFFT_FORWARD,
				1,
				&queue,
				0,
				NULL,
				NULL,
				&bufX,
				NULL,
				NULL);

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

  timings("without copy",n,T,N,MEDIAN);

  /* Release OpenCL memory objects. */
  clReleaseMemObject(bufX);

  free(X);

  /* Release the plan. */
  err = clfftDestroyPlan(&plan);

  /* Release clFFT library. */
  clfftTeardown();

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return 0;
}
  
