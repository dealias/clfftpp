#include <iostream>
#include <assert.h>
#include <clFFT.h> /* No need to explicitely include the OpenCL headers */
extern "C" {
#include "clutils.h"
}

class clfft_base
{
private:
  static int count_zero;
protected:
  clfftPlanHandle plan, backwardplan;
  cl_context ctx;
  cl_command_queue queue;
  unsigned int ncomplexfloats, nrealfloats;
  size_t inbuf_size, outbuf_size;
  clfftPrecision precision;
  bool realtocomplex, inplace;

  const char *clfft_errorstring(const cl_int err) {
    const char *errstring = NULL;
    
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
  
  virtual void set_nfloats() = 0;
  
  void set_buf_size() {
    set_nfloats();
    if(realtocomplex) {
      inbuf_size = nrealfloats
	* (precision == CLFFT_DOUBLE ? sizeof(double) : sizeof(float));
      outbuf_size = ncomplexfloats
	* (precision == CLFFT_DOUBLE ? sizeof(double) : sizeof(float));
    } else {
      inbuf_size = ncomplexfloats
	* (precision == CLFFT_DOUBLE ? sizeof(double) : sizeof(float));
      outbuf_size = inbuf_size;
    }
  }

public:
  cl_mem inbuf, outbuf;
  clfft_base() {
    if(count_zero == 0)
      clfft_setup();
    ++count_zero;
    precision = CLFFT_DOUBLE;
    //precision=CLFFT_SINGLE;
    realtocomplex = false;
    inplace = true;
  }

  ~clfft_base() {
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

  const unsigned int get_ncomplexfloats() {return ncomplexfloats;}
  const unsigned int get_nrealfloats() {return nrealfloats;}

  void create_clbuf(cl_mem *buf = NULL, size_t buf_size = 0) {
    cl_int ret;
    if(buf == NULL)
      buf = &inbuf;
    if(buf_size == 0)
      buf_size = inbuf_size;
    *buf = clCreateBuffer(ctx, 
			  CL_MEM_READ_WRITE, 
			  buf_size,
			  NULL,
			  &ret);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void inbuf_to_ram(double *X, cl_mem buf0,
		    const cl_uint nwait,
		    const cl_event *wait, cl_event *event) {
    cl_mem buf = (buf0 != NULL) ? buf0 : inbuf;
    cl_int ret;
    ret = clEnqueueReadBuffer(queue,
			      buf,
			      CL_TRUE,
			      0,
			      inbuf_size,
			      X,
			      nwait,
			      wait,
			      event);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }
  void inbuf_to_ram(double *X) {
    inbuf_to_ram(X, NULL, 0, NULL, NULL);
  }
  void inbuf_to_ram(double *X, cl_event *event) {
    inbuf_to_ram(X, NULL, 0, NULL, event);
  }
  void inbuf_to_ram(double *X,
		 const cl_uint nwait, const cl_event *wait) {
    inbuf_to_ram(X, NULL, nwait, wait, NULL);
  }
  void inbuf_to_ram(double *X, const cl_uint nwait,
		 const cl_event *wait, cl_event *event) {
    inbuf_to_ram(X, NULL, nwait, wait, event);
  }
  
  void outbuf_to_ram(double *X, cl_mem buf0,
		    const cl_uint nwait,
		    const cl_event *wait, cl_event *event) {
    cl_mem buf = (buf0 != NULL) ? buf0 : outbuf;
    cl_int ret;
    ret = clEnqueueReadBuffer(queue,
			      buf,
			      CL_TRUE,
			      0,
			      outbuf_size,
			      X,
			      nwait,
			      wait,
			      event);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }
  void outbuf_to_ram(double *X) {
    outbuf_to_ram(X, NULL, 0, NULL, NULL);
  }
  void outbuf_to_ram(double *X, cl_event *event) {
    outbuf_to_ram(X, NULL, 0, NULL, event);
  }
  void outbuf_to_ram(double *X,
		     const cl_uint nwait, const cl_event *wait) {
    outbuf_to_ram(X, NULL, nwait, wait, NULL);
  }
  void outbuf_to_ram(double *X, const cl_uint nwait,
		     const cl_event *wait, cl_event *event) {
    outbuf_to_ram(X, NULL, nwait, wait, event);
  }
  
  void ram_to_inbuf(const double *X, cl_mem buf0, 
		 const cl_uint nwait,
		 const cl_event *wait, cl_event *event) {
    cl_mem buf = (buf0 != NULL) ? buf0 : inbuf;
    cl_int ret;
    ret = clEnqueueWriteBuffer(queue,
			       buf,
			       CL_TRUE,
			       0,
			       inbuf_size,
			       X,
			       nwait,
			       wait,
			       event);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }
  void ram_to_inbuf(const double *X) {
    ram_to_inbuf(X, NULL, 0, NULL, NULL);
  }
  void ram_to_inbuf(const double *X, cl_event *event) {
    ram_to_inbuf(X, NULL, 0, NULL, event);
  }
  void ram_to_inbuf(const double *X, 
		    const cl_uint nwait, const cl_event *wait) {
    ram_to_inbuf(X, NULL, nwait, wait, NULL);
  }
  void ram_to_inbuf(const double *X, const cl_uint nwait,  
		 const cl_event *wait, cl_event *event) {
    ram_to_inbuf(X, NULL, nwait, wait, event);
  }

  void ram_to_outbuf(const double *X, cl_mem buf0, 
		 const cl_uint nwait,
		 const cl_event *wait, cl_event *event) {
    cl_mem buf = (buf0 != NULL) ? buf0 : outbuf;
    cl_int ret;
    ret = clEnqueueWriteBuffer(queue,
			       buf,
			       CL_TRUE,
			       0,
			       outbuf_size,
			       X,
			       nwait,
			       wait,
			       event);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }
  void ram_to_outbuf(const double *X) {
    ram_to_outbuf(X, NULL, 0, NULL, NULL);
  }
  void ram_to_outbuf(const double *X, cl_event *event) {
    ram_to_outbuf(X, NULL, 0, NULL, event);
  }
  void ram_to_outbuf(const double *X, 
		    const cl_uint nwait, const cl_event *wait) {
    ram_to_outbuf(X, NULL, nwait, wait, NULL);
  }
  void ram_to_outbuf(const double *X, const cl_uint nwait,  
		 const cl_event *wait, cl_event *event) {
    ram_to_outbuf(X, NULL, nwait, wait, event);
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
    cl_mem *buf0 = (inbuf0 != NULL) ? inbuf0 : &inbuf;
    cl_mem *buf1 = (outbuf0 != NULL) ? outbuf0 : &outbuf;
    cl_int ret;
    
    ret = clfftEnqueueTransform(plan, // clfftPlanHandle 	plHandle,
				direction, // direction
				1,  //cl_uint 	numQueuesAndEvents,
				&queue,
				nwait, // cl_uint 	numWaitEvents,
				wait, // const cl_event * 	waitEvents,
				done, // cl_event * 	outEvents,
				buf0, // cl_mem * 	inputBuffers,
				buf1, // cl_mem * 	outputBuffers,
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

  void set_nfloats() {
    ncomplexfloats = nx * 2;
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

    // FIXME: deal with out-of-place too.
    create_clbuf(&inbuf, inbuf_size);
  }

public:
  clfft1() {
    ctx = NULL;
    queue = NULL;
    inbuf = NULL;
    outbuf = NULL;
    nx = 0;
    set_buf_size();
  }

  clfft1(unsigned int nx0, cl_command_queue queue0, cl_context ctx0,
	 cl_mem inbuf0 = NULL, cl_mem outbuf0 = NULL) {
    nx = nx0;
    queue = queue0;
    ctx = ctx0;
    inbuf = inbuf0;
    outbuf = outbuf0;
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

  void set_nfloats() {
    ncomplexfloats = nx * ny * 2;
  }

  void setup() {
    set_buf_size();

    clfftDim dim = CLFFT_2D;
    size_t clLengths[2] = {nx, ny};

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

    // TODO: deal with out-of-place too.
    create_clbuf(&inbuf, inbuf_size);
  }

public:
  clfft2() {
    ctx = NULL;
    queue = NULL;
    inbuf = NULL;
    outbuf = NULL;
    nx = 0;
    set_buf_size();
  }

  clfft2(unsigned int nx0, unsigned int ny0, 
	 cl_command_queue queue0, cl_context ctx0,
	 cl_mem inbuf0 = NULL) {
    nx = nx0;
    ny = ny0;
    queue = queue0;
    ctx = ctx0;
    inbuf = inbuf0;
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

  void set_nfloats() {
    nrealfloats =  nx;
    ncomplexfloats =  2 * (1 + nx / 2);
  }

  void setup() {
    realtocomplex = true;
    inplace = false;

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
				 CLFFT_OUTOFPLACE);
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

    create_clbuf(&inbuf, inbuf_size);
    create_clbuf(&outbuf, outbuf_size);
  }

public:
  clfft1r() {
    ctx = NULL;
    queue = NULL;
    inbuf = NULL;
    outbuf = NULL;
    nx = 0;
    set_buf_size();
    realtocomplex = true;
    inplace = false;
  }

  clfft1r(unsigned int nx0, cl_command_queue queue0, cl_context ctx0,
	  cl_mem inbuf0 = NULL, cl_mem outbuf0 = NULL) {
    nx = nx0;
    queue = queue0;
    ctx = ctx0;
    inbuf = inbuf0;
    outbuf = outbuf0;
    setup();
    realtocomplex = true;
  }

  ~clfft1r() {
    cl_int ret;
    ret = clfftDestroyPlan(&plan);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }
};
