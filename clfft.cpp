#include <stdlib.h>

#include <iostream>
#include <timing.h>
#include <seconds.h>

/* No need to explicitely include the OpenCL headers */
#include <clFFT.h>

void show(float *X, int N)
{
  for(unsigned int i=0; i < N; ++i) {
    std::cout << "(" << X[2*i] << "," <<  X[2*i +1] << ")" << std::endl;
  }
}

void init(float *X, int N)
{
  for(unsigned int i=0; i < N; ++i) {
    X[2*i] = i;
    X[2*i + 1] = 0.0;
  }
}

int main() {
  cl_int err;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
  cl_context ctx = 0;
  cl_command_queue queue = 0;
  cl_mem bufX;
  float *X;
  cl_event event = NULL;
  int ret = 0;
  size_t N = 1024;
  N=262144;

  /* FFT library realted declarations */
  clfftPlanHandle planHandle;
  clfftDim dim = CLFFT_1D;
  size_t clLengths[1] = {N};

  /* Setup OpenCL environment. */
  err = clGetPlatformIDs( 1, &platform, NULL );
  err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL );

  props[1] = (cl_context_properties)platform;
  ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
  queue = clCreateCommandQueue( ctx, device, 0, &err );

  /* Setup clFFT. */
  clfftSetupData fftSetup;
  err = clfftInitSetupData(&fftSetup);
  err = clfftSetup(&fftSetup);


  /* Allocate host & initialize data. */
  /* Only allocation shown for simplicity. */
  X = (float *)malloc(N * 2 * sizeof(*X));

  int NT=1000;
  double *T=new double[NT];

  init(X,N);
  //show(X,N);
  
  /* Prepare OpenCL memory objects and place data inside them. */
  bufX = clCreateBuffer(ctx, 
			CL_MEM_READ_WRITE, N * 2 * sizeof(*X),
			NULL,
			&err);

  // Copy X to bufX
  err = clEnqueueWriteBuffer(queue,
			     bufX,
			     CL_TRUE,
			     0,
			     N * 2 * sizeof(*X),
			     X,
			     0,
			     NULL,
			     NULL);

  // Create a default plan for a complex FFT.
  err = clfftCreateDefaultPlan(&planHandle, 
			       ctx, 
			       dim, 
			       clLengths);

  /* Set plan parameters. */
  err = clfftSetPlanPrecision(planHandle, 
			      CLFFT_SINGLE);
  err = clfftSetLayout(planHandle, 
		       CLFFT_COMPLEX_INTERLEAVED, 
		       CLFFT_COMPLEX_INTERLEAVED);
  err = clfftSetResultLocation(planHandle, 
			       CLFFT_INPLACE);

  // Bake the plan.
  err = clfftBakePlan(planHandle, 
		      1, // numQueues: number of experiments 
		      &queue, // commQueueFFT
		      NULL, // Always NULL
		      NULL // Always NULL
		      );

  for(int i=0; i < NT; ++i) {
    init(X,N);
    seconds();
    // Copy X to bufX
    err = clEnqueueWriteBuffer(queue,
			       bufX,
			       CL_TRUE,
			       0,
			       N * 2 * sizeof(*X),
			       X,
			       0,
			       NULL,
			       NULL);

    // Execute the plan.
    err = clfftEnqueueTransform(planHandle,
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
			      N * 2 * sizeof( *X ), 
			      X, 
			      0, 
			      NULL, 
			      NULL );
    T[i]=seconds();
  }
  timings("with copy",N,T,NT,MEDIAN);

  /* Release OpenCL memory objects. */
  clReleaseMemObject(bufX);

  free(X);

  /* Release the plan. */
  err = clfftDestroyPlan(&planHandle);

  /* Release clFFT library. */
  clfftTeardown();

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return ret;
}
  
