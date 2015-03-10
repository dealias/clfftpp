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
  size_t rbuf_size, cbuf_size, var_size;
  clfftPrecision precision;
  bool realtocomplex, inplace;
  cl_mem workmem;

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
  
  void set_buf_size() {
    var_size = precision == CLFFT_DOUBLE ? sizeof(double) : sizeof(float);
    cbuf_size = ncomplex(-1) * 2 * var_size;
    if(realtocomplex) {
      if(inplace) 
	rbuf_size = ncomplex(-1) * 2 * var_size;
      else
	rbuf_size = nreal(-1) * var_size;
    } else {
      rbuf_size = 0;
    }
  }
public:
  clfft_base() {
    if(count_zero == 0)
      clfft_setup();
    ++count_zero;
    precision = CLFFT_DOUBLE;
    //precision = CLFFT_SINGLE;
    realtocomplex = false;
    inplace = true;
  }

  ~clfft_base() {
    --count_zero;
    if(count_zero == 0)
      clfftTeardown();
  }

  void set_workmem() {
    cl_int ret;

    size_t nwork = 0;
    ret = clfftGetTmpBufSize(plan, &nwork);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    if(nwork > 0) {
      workmem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nwork, 0, &ret);
      if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
      assert(ret == CL_SUCCESS);
    } else {
      workmem = NULL;
    }
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

  virtual const unsigned int nreal(const int dim) = 0;
  virtual const unsigned int ncomplex(const int dim) = 0;

  void create_rbuf(cl_mem *buf) {
    cl_int ret;
    *buf = clCreateBuffer(ctx,
			  CL_MEM_READ_WRITE,
			  rbuf_size,
			  NULL,
			  &ret);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void create_cbuf(cl_mem *buf) {
    cl_int ret;
    *buf = clCreateBuffer(ctx,
			  CL_MEM_READ_WRITE,
			  cbuf_size,
			  NULL,
			  &ret);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void ram_to_buf(const double *X, cl_mem *buf, const size_t buf_size,
		   const cl_uint nwait,
		   const cl_event *wait, cl_event *event) {

    cl_int ret;
    ret = clEnqueueWriteBuffer(queue,
			       *buf, // cl_mem buffer, 
			       CL_TRUE,// cl_bool blocking_read, 
			       0, // size_t offset
			       buf_size, // size_t cb
			       X, //  	void *ptr, 
			       nwait, // cl_uint num_events_in_wait_list, 
			       wait, // const cl_event *event_wait_list, 
			       event); // cl_event *event
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void ram_to_cbuf(const double *X, cl_mem *buf, 
		   const cl_uint nwait,
		   const cl_event *wait, cl_event *event) {
    ram_to_buf(X, buf, cbuf_size, nwait, wait, event);
  }

  void ram_to_rbuf(const double *X, cl_mem *buf, 
		   const cl_uint nwait,
		   const cl_event *wait, cl_event *event) {
    ram_to_buf(X, buf, rbuf_size, nwait, wait, event);
  }

  void buf_to_ram(double *X, cl_mem *buf, const size_t buf_size,
		  const cl_uint nwait,
		  const cl_event *wait, cl_event *event) {
    cl_int ret;
    ret = clEnqueueReadBuffer(queue,
			      *buf,
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

  void cbuf_to_ram(double *X, cl_mem *buf,
		  const cl_uint nwait,
		  const cl_event *wait, cl_event *event) {
    buf_to_ram(X, buf, cbuf_size, nwait, wait, event);
  }

  // FIXME: add overloaded operators to deal with events or no.

  void rbuf_to_ram(double *X, cl_mem *buf,
		  const cl_uint nwait,
		  const cl_event *wait, cl_event *event) {
    buf_to_ram(X, buf, rbuf_size, nwait, wait, event);
  }

  void finish() {
    cl_int ret = clFinish(queue);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void transform(clfftDirection direction, 
		 cl_mem *inbuf, cl_mem *outbuf,
		 cl_uint nwait = 0, cl_event *wait = NULL, 
		 cl_event *done = NULL) {
    cl_int ret;
    ret = clfftEnqueueTransform(plan, // clfftPlanHandle 	plHandle,
				direction, // direction
				1,  //cl_uint 	numQueuesAndEvents,
				&queue,
				nwait, // cl_uint 	numWaitEvents,
				wait, // const cl_event * 	waitEvents,
				done, // cl_event * 	outEvents,
				inbuf, // cl_mem * 	inputBuffers,
				outbuf, // cl_mem * 	outputBuffers,
				workmem // cl_mem 	tmpBuffer 
				);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }
  
  void forward(cl_mem *inbuf, cl_mem *outbuf,
	       cl_uint nwait = 0, cl_event *wait = NULL, 
	       cl_event *done = NULL) {
    transform(CLFFT_FORWARD, inbuf, outbuf, nwait, wait, done);
  }

  // void forward(cl_mem *inbuf, cl_uint nwait, cl_event *wait, cl_event *done) {
  //   forward(NULL, NULL, nwait, wait, done);
  // }
  
  void backward(cl_mem *inbuf0, cl_mem *outbuf0,
  		cl_uint nwait = 0, cl_event *wait = NULL, 
  		cl_event *done = NULL) {
    transform(CLFFT_BACKWARD, inbuf0, outbuf0, nwait, wait, done);
  }
  
  // void backward(cl_uint nwait, cl_event *wait, cl_event *done) {
  //   backward(NULL, NULL, nwait, wait, done);
  // }

  // void backward(cl_event *done) {
  //   backward(NULL, NULL, 0, NULL, done);
  // }
  
};

class clfft1 : public clfft_base
{
private:
  unsigned nx; // size of problem

  void setup() {
    realtocomplex = false;
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
				 inplace ? CLFFT_INPLACE : CLFFT_OUTOFPLACE);
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

    set_workmem();
  }

public:
  clfft1() {
    ctx = NULL;
    queue = NULL;
    nx = 0;
    set_buf_size();
    inplace = true;
  }

  clfft1(unsigned int nx0, bool inplace0, 
	 cl_command_queue queue0, cl_context ctx0) {
    nx = nx0;
    queue = queue0;
    ctx = ctx0;
    inplace = inplace0;
    setup();
  }

  ~clfft1() {
    cl_int ret;
    ret = clfftDestroyPlan(&plan);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  const unsigned int nreal(const int dim = -1) {return 0;}
  const unsigned int ncomplex(const int dim = -1) {
    switch(dim) {
    case -1:
      return nx;
      break;
    case 0:
      return nx;
      break;
    default:
      std::cerr << dim
		<< "is an invalid dimension for clfft1::ncomplex"
		<< std::endl;
      exit(1);
      return 0;
    }
  }

  virtual const size_t complex_buf_size() {
    return nx * 2 * var_size;
  }

};

class clfft2 : public clfft_base
{
private:
  unsigned nx, ny; // size of problem

  void setup() {
    realtocomplex = false;
    set_buf_size();

    clfftDim dim = CLFFT_2D;
    //size_t clLengths[2] = {nx, ny};
    size_t clLengths[2] = {ny, nx}; // They lied when they said it was row-major

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
				 inplace ? CLFFT_INPLACE : CLFFT_OUTOFPLACE);
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

    set_workmem();
  }

public:
  clfft2() {
    ctx = NULL;
    queue = NULL;
    inplace = true;
    nx = 0;
    set_buf_size();
  }

  clfft2(unsigned int nx0, unsigned int ny0, bool inplace0,
	 cl_command_queue queue0, cl_context ctx0) {
    nx = nx0;
    ny = ny0;
    inplace = inplace0;
    queue = queue0;
    ctx = ctx0;
    setup();
  }

  ~clfft2() {
    cl_int ret;
    ret = clfftDestroyPlan(&plan);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
  }

  const unsigned int nreal(const int dim = -1) {return 0;}
  const unsigned int ncomplex(const int dim = -1) {
    switch(dim) {
    case -1:
      return nx * ny;
    case 0:
      return nx;
    case 1:
      return ny;
    default:
      std::cerr << dim 
		<< "is an invalid dimension for clfft2::ncomplex"
		<< std::endl;
      exit(1);
    }
    return 0;
  }


  virtual const unsigned int complex_buf_size(const int dim) {
    return nx * ny * 2 * var_size;
  }
};

class clfft1r : public clfft_base
{
private:
  unsigned nx; // size of problem for clFFT

  void setup() {
    realtocomplex = true;

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
				 inplace ? CLFFT_INPLACE : CLFFT_OUTOFPLACE);
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

    set_workmem();
  }

public:
  clfft1r() {
    ctx = NULL;
    queue = NULL;
    nx = 0;
    set_buf_size();
  }

  clfft1r(unsigned int nx0, bool inplace0, 
	  cl_command_queue queue0, cl_context ctx0) {
    nx = nx0;
    queue = queue0;
    ctx = ctx0;
    inplace = inplace0;

    setup();
  }

  ~clfft1r() {
    cl_int ret;
    ret = clfftDestroyPlan(&plan);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  const unsigned int ncomplex(const int dim = -1) {
    switch(dim) {
    case -1:
      return 1 + nx / 2;
    case 0:
      return 1 + nx / 2;
    default:
      std::cerr << dim 
		<< "is an invalid dimension for clfft1r::ncomplex"
		<< std::endl;
      exit(1);
    }
    return 0;
  }

  virtual const unsigned int nreal(const int dim = -1) {
    switch(dim) {
    case -1:
      return nx;
    case 0:
      return nx;
    default:
      std::cerr << dim 
		<< "is an invalid dimension for clfft1r::nreal"
		<< std::endl;
      exit(1);
    }
    return 0;
  }

  size_t complex_buf_size() {
    return  (1 + nx / 2) * 2 * var_size;
  }

  size_t real_buf_size() {
    return nx * var_size;
  }

};

class clfft2r : public clfft_base
{
private:
  unsigned nx, ny; // size of problem for clFFT

  void setup() {
    realtocomplex = true;
    inplace = false;

    set_buf_size();
  
    clfftDim dim = CLFFT_2D;
    //size_t clLengths[2] = {nx, ny};
    size_t clLengths[2] = {ny, nx}; // They lied when they said it was row-major

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
				 inplace ? CLFFT_INPLACE : CLFFT_OUTOFPLACE);
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

    set_workmem();
  }

public:
  clfft2r() {
    ctx = NULL;
    queue = NULL;
    nx = 0;
    set_buf_size();
  }

  clfft2r(unsigned int nx0, unsigned int ny0, bool inplace0, 
	  cl_command_queue queue0, cl_context ctx0) {
    nx = nx0;
    ny = ny0;
    queue = queue0;
    ctx = ctx0;
    inplace = inplace0;
    setup();
  }

  ~clfft2r() {
    cl_int ret;
    ret = clfftDestroyPlan(&plan);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  const unsigned int ncomplex(const int dim = -1) {
    switch(dim) {
    case -1:
      //return (1 + ny / 2) * nx;
      return nx * ny + 100; // FIXME: WTF?
    case 0:
      return nx;
    case 1:      
      return 1 + ny / 2;
    default:
      std::cerr << dim 
		<< "is an invalid dimension for clfft2r::ncomplex"
		<< std::endl;
      exit(1);
    }
    return 0;
  }

  const unsigned int nreal(const int dim = -1) {
    switch(dim) {
    case -1:
      if(inplace)
	return ncomplex(-1);
      else
	return nx * ny;
    case 0:
      return nx;
    case 1:
      return ny;
    default:
      std::cerr << dim 
		<< "is an invalid dimension for clfft2r::ncomplex"
		<< std::endl;
      exit(1);
    }
    return 0;
  }
};
