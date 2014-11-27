/* No need to explicitely include the OpenCL headers */
#include <clFFT.h>
#include <iostream>
#include <clutils.h>

#include <assert.h>

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

  const char* clfft_errorstring(const cl_int err) {
    const char* errstring = NULL;
    
    errstring = clErrorString(err);

    switch (err) {
    case CL_SUCCESS:
      errstring = "Success";
      break;
    case CLFFT_BUGCHECK:
      errstring = "CLFFT_BUGCHECK";
      break;
    case CLFFT_NOTIMPLEMENTED:
      errstring = "CLFFT_NOTIMPLEMENTED";
      break;
    case CLFFT_TRANSPOSED_NOTIMPLEMENTED:
      errstring = "CLFFT_TRANSPOSED_NOTIMPLEMENTED";
      break;
    case CLFFT_FILE_NOT_FOUND:
      errstring = "CLFFT_FILE_NOT_FOUND:";
      break;
    case CLFFT_FILE_CREATE_FAILURE:
      errstring = "CLFFT_FILE_CREATE_FAILURE";
      break;
    case CLFFT_VERSION_MISMATCH:
      errstring = "CLFFT_VERSION_MISMATCH";
      break;
    case CLFFT_INVALID_PLAN:
      errstring = "CLFFT_INVALID_PLAN";
      break;
    case CLFFT_DEVICE_NO_DOUBLE: 
      errstring = "CLFFT_DEVICE_NO_DOUBLE";
      break;
    }
    return errstring;
  }

public:
  clfft_base(){
    if(count_zero == 0)
      clfft_setup();
    ++count_zero;
    precision = CLFFT_DOUBLE;
    //precision=CLFFT_SINGLE;
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
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
    ret = clfftSetup(&fftSetup);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }


  double * create_rambuf() {
    return new double[buf_size];
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

  void cl_to_ram(double *X, cl_mem bufX0, 
		 const cl_uint nwait,
		 const cl_event *wait, cl_event *event) {
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

  void cl_to_ram(double *X) {
    cl_to_ram(X, NULL, 0, NULL, NULL);
  }
  
  void cl_to_ram(double *X,
		 const cl_uint nwait,
		 const cl_event *wait) {
    cl_to_ram(X, NULL, nwait, wait, NULL);
  }
  
  void cl_to_ram(double *X, cl_event *event) {
    cl_to_ram(X, NULL, 0, NULL, event);
  }
  
  void cl_to_ram(double *X,
		 const cl_uint nwait,
		 const cl_event *wait, cl_event *event) {
    cl_to_ram(X, NULL, nwait, wait, event);
  }
  
  void ram_to_cl(double *X, cl_mem bufX0, 
		 const cl_uint nwait,
		 const cl_event *wait, cl_event *event) {
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

  // with no events
  void ram_to_cl(double *X) {
    ram_to_cl(X, NULL, 0, NULL, NULL);
  }

  // with event
  void ram_to_cl(double *X,  
		 cl_event *event) {
    ram_to_cl(X, NULL, 0, NULL, event);
  }

  // with wait event(s)
  void ram_to_cl(double *X,  
		 const cl_uint nwait,
		 const cl_event *wait) {
    ram_to_cl(X, NULL, nwait, wait, NULL);
  }

  // with wait event(s) and new event
  void ram_to_cl(double *X,  
		 const cl_uint nwait,  const cl_event *wait, 
		 cl_event *event) {
    ram_to_cl(X, NULL, nwait, wait, event);
  }

  void finish() {
    cl_int ret = clFinish(queue);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void transform(clfftDirection direction, 
		 cl_mem *inbuf0 = NULL, cl_mem *outbuf0 = NULL,
		 cl_uint nwait = 0, cl_event *wait = NULL, 
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
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }
  
  void forward(cl_mem *inbuf0 = NULL, cl_mem *outbuf0 = NULL,
		       cl_uint nwait = 0, cl_event *wait = NULL, 
		       cl_event *done = NULL) {
    transform(CLFFT_FORWARD, inbuf0, outbuf0, nwait, wait, done);
  }

  void forward(cl_uint nwait, cl_event *wait, cl_event *done) {
    forward(NULL, NULL, nwait, wait, done);
  }
  
  void backward(cl_mem *inbuf0 = NULL, cl_mem *outbuf0 = NULL,
		       cl_uint nwait = 0, cl_event *wait = NULL, 
		       cl_event *done = NULL) {
    transform(CLFFT_BACKWARD, inbuf0, outbuf0, nwait, wait, done);
  }
  
  void backward(cl_uint nwait, cl_event *wait, cl_event *done) {
    backward(NULL, NULL, nwait, wait, done);
  }
  
};

class clfft1 : public clfft_base
{
private:
  unsigned nx; // size of problem

  void set_buf_size() {
    buf_size = nx * 2 * sizeof(double); // TODO: variable precision
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
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetPlanPrecision(plan, 
				precision);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetLayout(plan,
			 CLFFT_COMPLEX_INTERLEAVED, 
			 CLFFT_COMPLEX_INTERLEAVED);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetResultLocation(plan, 
				 CLFFT_INPLACE);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftBakePlan(plan,
			1, // numQueues: number of experiments 
			&queue, // commQueueFFT
			NULL, // Always NULL
			NULL // Always NULL
			);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
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
    buf_size = nx *ny * 2 * sizeof(double); // TODO: variable precision
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
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetPlanPrecision(plan, 
				precision);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetLayout(plan, 
			 CLFFT_COMPLEX_INTERLEAVED, 
			 CLFFT_COMPLEX_INTERLEAVED);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetResultLocation(plan, 
				 CLFFT_INPLACE);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftBakePlan(plan,
			1, // numQueues: number of experiments 
			&queue, // commQueueFFT
			NULL, // Always NULL
			NULL // Always NULL
			);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
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
  unsigned nx; // size of problem for clFFT

  void set_buf_size() {
    buf_size = (nx + 1) * 2 * sizeof(double); // TODO: variable precision
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
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetPlanPrecision(plan, 
				precision);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetLayout(plan, 
			 CLFFT_REAL,
			 CLFFT_HERMITIAN_INTERLEAVED);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetResultLocation(plan, 
				 CLFFT_INPLACE);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftBakePlan(plan,
			1, // numQueues: number of experiments 
			&queue, // commQueueFFT
			NULL, // Always NULL
			NULL // Always NULL
			);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
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
	 cl_mem bufX0 = NULL) {
    nx = nx0;
    queue = queue0;
    ctx = ctx0;
    bufX = bufX0;
    setup();
  }

  ~clfft1r() {
    cl_int ret;
    ret = clfftDestroyPlan(&plan);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }


};
