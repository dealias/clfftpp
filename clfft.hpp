/* No need to explicitely include the OpenCL headers */
#include <clFFT.h>

void clfft_setup();

class clfft_base
{
private:
  static int count_zero;
public:
  clfft_base(){
    if(count_zero == 0)
      clfft_setup();
    ++count_zero;
  }

  ~clfft_base(){
    --count_zero;
    if(count_zero == 0)
      clfftTeardown();
  }

  void clfft_setup() {
    cl_int err;  
    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    err = clfftSetup(&fftSetup);
  }
};

class clfft1 : public clfft_base
{
private:
  clfftPlanHandle plan;
  unsigned nx; // size of problem
  cl_mem bufX;
  
  // TODO: should be in base class?
  cl_command_queue queue;
  cl_context ctx;
public:
  clfft1(unsigned int nx0, cl_command_queue queue0, cl_context ctx0,
	 cl_mem bufX0) {
    nx=nx0;
    queue=queue0;
    ctx=ctx0;
    bufX=bufX0;
    
    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = {nx};

    cl_int err;
    err = clfftCreateDefaultPlan(&plan, 
				 ctx, 
				 dim, 
				 clLengths);
    err = clfftSetPlanPrecision(plan, 
				CLFFT_SINGLE);
    err = clfftSetLayout(plan, 
			 CLFFT_COMPLEX_INTERLEAVED, 
			 CLFFT_COMPLEX_INTERLEAVED);
    err = clfftSetResultLocation(plan, 
				 CLFFT_INPLACE);
    err = clfftBakePlan(plan,
			1, // numQueues: number of experiments 
			&queue, // commQueueFFT
			NULL, // Always NULL
			NULL // Always NULL
			);
  }

  ~clfft1() {
    cl_int err;
    err = clfftDestroyPlan(&plan);
  }
  void fft() {
    // FIXME: compute FFTs and such.
    cl_int err;
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


  }

};
