/* No need to explicitely include the OpenCL headers */
#include <clFFT.h>
#include <iostream>
#include <clutils.h>

#include <assert.h>

typedef float REAL;

void clfft_setup();

class clfft_base
{
private:
  static int count_zero;
protected:
  clfftPlanHandle plan;
  cl_context ctx;
  cl_command_queue queue;
  cl_mem bufX;
  int buf_size;
  clfftPrecision precision;

public:
  clfft_base(){
    if(count_zero == 0)
      clfft_setup();
    ++count_zero;
    //precision=CLFFT_DOUBLE;
    precision=CLFFT_SINGLE;
  }

  ~clfft_base(){
    --count_zero;
    if(count_zero == 0)
      clfftTeardown();
  }

  void clfft_setup() {
    cl_int ret;  
    clfftSetupData fftSetup;
    ret = clfftInitSetupData(&fftSetup);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
    ret = clfftSetup(&fftSetup);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }


  REAL * create_rambuf() {
    return new REAL[buf_size];
  }
 
  cl_mem create_clbuf() {
    cl_int ret;
    bufX = clCreateBuffer(ctx, 
			  CL_MEM_READ_WRITE, 
			  buf_size,
			  NULL,
			  &ret);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
    return bufX;
  }

  void cl_to_ram(REAL *X, cl_mem bufX0=NULL, 
		 const cl_uint nwait=0,
		 const cl_event *wait=NULL, cl_event *event=NULL) {
    cl_mem buf = (bufX0 != NULL) ? bufX0 : bufX;
    cl_int ret;
    ret = clEnqueueReadBuffer(queue,
			      buf,
			      CL_TRUE,
			      0,
			      buf_size,
			      X,
			      nwait,
			      wait,
			      event);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void ram_to_cl(REAL *X, cl_mem bufX0=NULL, 
		 const cl_uint nwait=0,
		 const cl_event *wait=NULL, cl_event *event=NULL) {
    cl_mem buf = (bufX0 != NULL) ? bufX0 : bufX;
    cl_int ret;
    ret = clEnqueueWriteBuffer(queue,
			       buf,
			       CL_TRUE,
			       0,
			       buf_size,
			       X,
			       nwait,
			       wait,
			       event);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void wait() {
    cl_int ret = clFinish(queue);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void transform(clfftDirection direction, 
		 cl_mem *inbuf0 = NULL, cl_mem *outbuf0 = NULL,
		 cl_event *wait = NULL, cl_uint nwait = 0,
		 cl_event *done = NULL) {
    cl_mem *inbuf = (inbuf0 != NULL) ? inbuf0 : &bufX;
    cl_int ret;
    
    ret = clfftEnqueueTransform(plan, // clfftPlanHandle 	plHandle,
				direction,// direction
				1,  //cl_uint 	numQueuesAndEvents,
				&queue,
				nwait, // cl_uint 	numWaitEvents,
				wait, // const cl_event * 	waitEvents,
				done, // cl_event * 	outEvents,
				inbuf, // cl_mem * 	inputBuffers,
				outbuf0, // cl_mem * 	outputBuffers,
				NULL // cl_mem 	tmpBuffer 
				);
    // FIXME: add events.
    // FIXME: move into base class (all calls are basically the same).
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }
  
  virtual void forward(cl_mem *inbuf0 = NULL, cl_mem *outbuf0 = NULL,
		       cl_event *wait = NULL, cl_uint nwait = 0,
		       cl_event *done = NULL) {
    transform(CLFFT_FORWARD, inbuf0, outbuf0, wait, nwait, done);
  }

  virtual void backward(cl_mem *inbuf0 = NULL, cl_mem *outbuf0 = NULL,
		       cl_event *wait = NULL, cl_uint nwait = 0,
		       cl_event *done = NULL) {
    transform(CLFFT_BACKWARD, inbuf0, outbuf0, wait, nwait, done);
  }

  // virtual void forward(cl_mem bufX0=NULL) {
  //   cl_mem buf = (bufX0 != NULL) ? bufX0 : bufX;
  //   cl_int ret;
  //   ret = clfftEnqueueTransform(plan, // clfftPlanHandle 	plHandle,
  // 				CLFFT_FORWARD,// direction
  // 				1,  //cl_uint 	numQueuesAndEvents,
  // 				&queue,
  // 				0, // cl_uint 	numWaitEvents,
  // 				NULL, // const cl_event * 	waitEvents,
  // 				NULL, // cl_event * 	outEvents,
  // 				&buf, // cl_mem * 	inputBuffers,
  // 				NULL, // cl_mem * 	outputBuffers,
  // 				NULL // cl_mem 	tmpBuffer 
  // 				);
  //   // FIXME: add events.
  //   // FIXME: move into base class (all calls are basically the same).
  //   if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
  //   assert(ret == CL_SUCCESS);
  // }


};

class clfft1 : public clfft_base
{
private:
  unsigned nx; // size of problem

  void set_buf_size() {
    buf_size = nx * 2 * sizeof(REAL); // TODO: variable precision
  }

  void setup() {
    set_buf_size();

    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = {nx};

    cl_int ret;
    ret = clfftCreateDefaultPlan(&plan, 
				 ctx, 
				 dim, 
				 clLengths);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetPlanPrecision(plan, 
				precision);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetLayout(plan,
			 CLFFT_COMPLEX_INTERLEAVED, 
			 CLFFT_COMPLEX_INTERLEAVED);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetResultLocation(plan, 
				 CLFFT_INPLACE);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftBakePlan(plan,
			1, // numQueues: number of experiments 
			&queue, // commQueueFFT
			NULL, // Always NULL
			NULL // Always NULL
			);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

public:
  clfft1() {
    ctx = NULL;
    queue = NULL;
    bufX = NULL;
    nx = 0;
    set_buf_size();
  }

  clfft1(unsigned int nx0, cl_command_queue queue0, cl_context ctx0,
	 cl_mem bufX0=NULL) {
    nx=nx0;
    queue=queue0;
    ctx=ctx0;
    bufX=bufX0;
    setup();
  }

  ~clfft1() {
    cl_int ret;
    ret = clfftDestroyPlan(&plan);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

};

class clfft2 : public clfft_base
{
private:
  unsigned nx, ny; // size of problem

  void set_buf_size() {
    buf_size = nx *ny * 2 * sizeof(REAL); // TODO: variable precision
  }

  void setup() {
    set_buf_size();

    clfftDim dim = CLFFT_2D;
    size_t clLengths[2] = {nx,ny};

    cl_int ret;
    ret = clfftCreateDefaultPlan(&plan, 
				 ctx, 
				 dim, 
				 clLengths);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetPlanPrecision(plan, 
				precision);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetLayout(plan, 
			 CLFFT_COMPLEX_INTERLEAVED, 
			 CLFFT_COMPLEX_INTERLEAVED);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetResultLocation(plan, 
				 CLFFT_INPLACE);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftBakePlan(plan,
			1, // numQueues: number of experiments 
			&queue, // commQueueFFT
			NULL, // Always NULL
			NULL // Always NULL
			);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

public:
  clfft2() {
    ctx = NULL;
    queue = NULL;
    bufX = NULL;
    nx = 0;
    set_buf_size();
  }

  clfft2(unsigned int nx0, unsigned int ny0, 
	 cl_command_queue queue0, cl_context ctx0,
	 cl_mem bufX0=NULL) {
    nx=nx0;
    ny=ny0;
    queue=queue0;
    ctx=ctx0;
    bufX=bufX0;
    setup();
  }

  ~clfft2() {
    cl_int ret;
    ret = clfftDestroyPlan(&plan);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
  }

};


class clfft1r : public clfft_base
{
private:
  unsigned nx; // size of problem

  void set_buf_size() {
    buf_size = (nx + 1) * 2 * sizeof(REAL); // TODO: variable precision
  }

  void setup() {
    set_buf_size();

    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = {nx};

    cl_int ret;
    ret = clfftCreateDefaultPlan(&plan, 
				 ctx, 
				 dim, 
				 clLengths);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetPlanPrecision(plan, 
				precision);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetLayout(plan, 
			 CLFFT_COMPLEX_INTERLEAVED, 
			 CLFFT_REAL);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetResultLocation(plan, 
				 CLFFT_INPLACE);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftBakePlan(plan,
			1, // numQueues: number of experiments 
			&queue, // commQueueFFT
			NULL, // Always NULL
			NULL // Always NULL
			);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }
public:
  clfft1r() {
    ctx = NULL;
    queue = NULL;
    bufX = NULL;
    nx = 0;
    set_buf_size();
  }

  clfft1r(unsigned int nx0, cl_command_queue queue0, cl_context ctx0,
	 cl_mem bufX0=NULL) {
    nx=nx0;
    queue=queue0;
    ctx=ctx0;
    bufX=bufX0;
    setup();
  }

  ~clfft1r() {
    cl_int ret;
    ret = clfftDestroyPlan(&plan);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }


};
