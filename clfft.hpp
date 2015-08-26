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
  clfftPlanHandle forward_plan, backward_plan;
  cl_context ctx;
  cl_command_queue queue;
  size_t rbuf_size, cbuf_size;
  bool inplace, realtocomplex;
  clfftPrecision precision;
  size_t realsize;
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
  
  void create_default_plan(clfftPlanHandle &plan,  
			   clfftDim dim, size_t *clLengths) {
    cl_int ret;
    ret = clfftCreateDefaultPlan(&plan,
				 ctx,
				 dim,
				 clLengths);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void set_inout_place(clfftPlanHandle &plan) {
    cl_int ret;
    ret = clfftSetResultLocation(plan, 
				 inplace ? CLFFT_INPLACE : CLFFT_OUTOFPLACE);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void set_precision(clfftPlanHandle &plan, clfftPrecision precision) {
    cl_int ret;
    ret = clfftSetPlanPrecision(plan,
				precision);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
    if(precision == CLFFT_DOUBLE)
      realsize = sizeof(double);
    if(precision == CLFFT_SINGLE)
      realsize = sizeof(float);
  }

  void set_strides(clfftPlanHandle &plan, clfftDim dim,
		   size_t *istride, size_t *ostride) {
    cl_int ret;
    
    ret = clfftSetPlanInStride(plan,
			       dim, //const clfftDim  	dim,
			       istride //size_t * clStrides 
			       );
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftSetPlanOutStride(plan,
				dim, //const clfftDim  	dim,
				ostride //size_t * clStrides 
				);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void set_dists(clfftPlanHandle &plan, clfftDim dim,
		 size_t idist, size_t odist) {
    cl_int ret;
    ret = clfftSetPlanDistance(plan,
			       idist,
			       odist);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  
  void set_data_layout_complex(clfftPlanHandle &plan) {
    cl_int ret;

    ret = clfftSetLayout(plan,
			 CLFFT_COMPLEX_INTERLEAVED, 
			 CLFFT_COMPLEX_INTERLEAVED);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void set_data_layout_real_to_complex(clfftPlanHandle &plan) {
    cl_int ret;
    ret = clfftSetLayout(plan, 
			 CLFFT_REAL,
			 CLFFT_HERMITIAN_INTERLEAVED);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void set_data_layout_complex_to_real(clfftPlanHandle &plan) {
    cl_int ret;
    ret = clfftSetLayout(plan, 
			 CLFFT_HERMITIAN_INTERLEAVED,
			 CLFFT_REAL);
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void set_data_layout(clfftPlanHandle &plan, bool forward = true) {
    if(!realtocomplex) {
      set_data_layout_complex(plan);
    } else {
      if(forward) {
	set_data_layout_real_to_complex(plan);
      } else {
	set_data_layout_complex_to_real(plan);
      }
    }
  }

  void set_batchsize(clfftPlanHandle &plan, const size_t M) {
    cl_int ret;
    ret = clfftSetPlanBatchSize(plan, M); 
    if(ret != CL_SUCCESS) std::cerr << clfft_errorstring(ret) << std::endl;
    assert(ret == CL_SUCCESS);	
  }

  void set_buf_size() {
    cbuf_size = ncomplex(-1) * 2 * realsize;
    if(realtocomplex) {
      rbuf_size = inplace ? cbuf_size : nreal(-1) * realsize;
    } else {
      rbuf_size = 0;
    }
  }

  void bake_plan(clfftPlanHandle &plan) {
    cl_int ret;   
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
  clfft_base() :
    inplace(true), realtocomplex(false), precision(CLFFT_DOUBLE) {
    realsize = sizeof(double);
    if(count_zero++ == 0)
      clfft_setup();
    //precision = CLFFT_SINGLE;
  }

  clfft_base(cl_context ctx, cl_command_queue queue,
	     bool inplace, bool realtocomplex,
	     clfftPrecision precision = CLFFT_DOUBLE) :
    ctx(ctx), queue(queue), 
    inplace(inplace), realtocomplex(realtocomplex), precision(precision) {
    realsize = (precision == CLFFT_DOUBLE) ? sizeof(double) : sizeof(float);
    if(count_zero++ == 0)
      clfft_setup();
  }

  ~clfft_base() {
    cl_int ret;

    ret = clfftDestroyPlan(&forward_plan);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);

    ret = clfftDestroyPlan(&backward_plan);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
    
    if(--count_zero == 0) 
      clfftTeardown();
  }

  void set_workmem(clfftPlanHandle &plan) {
    cl_int ret;

    size_t nwork = 0; // FIXME: set up for free as well.
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

  void create_rbuf(cl_mem *buf, const int nreal = 0) {
    size_t n = (nreal == 0) ? rbuf_size : nreal * realsize;
    cl_int ret;
    *buf = clCreateBuffer(ctx,
			  CL_MEM_READ_WRITE,
			  n,
			  NULL,
			  &ret);
    if(ret != CL_SUCCESS) std::cerr << clErrorString(ret) << std::endl;
    assert(ret == CL_SUCCESS);
  }

  void create_cbuf(cl_mem *buf, const int ncomp = 0) {
    size_t n = (ncomp == 0) ? cbuf_size : 2 * ncomp * realsize;
    cl_int ret;
    *buf = clCreateBuffer(ctx,
			  CL_MEM_READ_WRITE,
			  n,
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
    //std::cout << "rbuf_size/sizeof(double): " << rbuf_size/sizeof(double) << std::endl;
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
    clfftPlanHandle plan 
      = (direction == CLFFT_FORWARD) ? forward_plan : backward_plan;
    
    cl_int ret;
    ret = clfftEnqueueTransform(plan, // clfftPlanHandle plHandle,
				direction, //direction
				1,  //cl_uint numQueuesAndEvents,
				&queue,
				nwait, // cl_uint numWaitEvents,
				wait, // const cl_event * waitEvents,
				done, // cl_event * outEvents,
				inbuf, // cl_mem * inputBuffers,
				outbuf, // cl_mem * outputBuffers,
				workmem // cl_mem tmpBuffer 
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

    setup_plan(forward_plan);
    setup_plan(backward_plan);
  }

  void setup_plan(clfftPlanHandle &plan) {
    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = {nx};

    create_default_plan(plan, dim, clLengths);
    set_precision(plan, precision);
    set_inout_place(plan);
    set_data_layout(plan);

    bake_plan(plan);
    set_workmem(plan);
  }

public:
  clfft1() : clfft_base(), nx(0) {
    set_buf_size();
  }

  clfft1(unsigned int nx, bool inplace, 
	 cl_command_queue queue, cl_context ctx) :
    clfft_base(ctx, queue, inplace, true, CLFFT_DOUBLE), nx(nx) {
    setup();
  }

  ~clfft1() {
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
		<< " is an invalid dimension for clfft1::ncomplex"
		<< std::endl;
      exit(1);
      return 0;
    }
  }

  virtual const size_t complex_buf_size() {
    return nx * 2 * realsize;
  }

};

class clfft2 : public clfft_base
{
private:
  unsigned nx, ny; // size of problem

  void setup() {
    realtocomplex = false;
    set_buf_size();

    setup_plan(forward_plan);
    setup_plan(backward_plan);
  }

  void setup_plan(clfftPlanHandle &plan) {

    clfftDim dim = CLFFT_2D;
    //size_t clLengths[2] = {nx, ny};
    size_t clLengths[2] = {ny, nx}; // They lied when they said it was row-major

    create_default_plan(plan, dim, clLengths);
    set_precision(plan, precision);
    set_data_layout(plan);
    set_inout_place(plan);

    bake_plan(plan);

    set_workmem(plan);
  }

public:
  clfft2() :
    clfft_base(), nx(0), ny(0) {
    set_buf_size();
  }

  clfft2(unsigned int nx, unsigned int ny, bool inplace,
	 cl_command_queue queue, cl_context ctx) :
    clfft_base(ctx, queue, inplace, true, CLFFT_DOUBLE), nx(nx), ny(ny) {
    setup();
  }

  ~clfft2() {
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
		<< " is an invalid dimension for clfft2::ncomplex"
		<< std::endl;
      exit(1);
    }
    return 0;
  }
  
  virtual const unsigned int complex_buf_size(const int dim) {
    return nx * ny * 2 * realsize;
  }
};

class clfft3 : public clfft_base
{
private:
  unsigned nx, ny, nz; // size of problem

  void setup() {
    realtocomplex = false;
    set_buf_size();

    setup_plan(forward_plan);
    setup_plan(backward_plan);
  }

  void setup_plan(clfftPlanHandle &plan) {

    clfftDim dim = CLFFT_3D;
    // They lied when they said it was row-major
    size_t clLengths[3] = {nz, ny, nx}; 

    create_default_plan(plan, dim, clLengths);
    set_precision(plan, precision);
    set_data_layout(plan);
    set_inout_place(plan);

    bake_plan(plan);

    set_workmem(plan);
  }

public:
  clfft3(): clfft_base(), nx(0), ny(0), nz(0) {
    set_buf_size();
  }

  clfft3(unsigned int nx, unsigned int ny, unsigned int nz, bool inplace,
	 cl_command_queue queue, cl_context ctx) :
    clfft_base(ctx, queue, inplace, true, CLFFT_DOUBLE), nx(nx), ny(ny), nz(nz){
    setup();
  }

  ~clfft3() {
  }

  const unsigned int nreal(const int dim = -1) {return 0;}
  const unsigned int ncomplex(const int dim = -1) {
    switch(dim) {
    case -1:
      return nx * ny * nz;
    case 0:
      return nx;
    case 1:
      return ny;
    case 2:
      return nz;
    default:
      std::cerr << dim 
		<< " is an invalid dimension for clfft2::ncomplex"
		<< std::endl;
      exit(1);
    }
    return 0;
  }

  virtual const unsigned int complex_buf_size(const int dim) {
    return nx * ny * nz * 2 * realsize;
  }
};


class clmfft1 : public clfft_base
{
private: 
  unsigned int nx;
  unsigned int M;
  unsigned int istride;
  unsigned int ostride;
  unsigned int idist;
  unsigned int odist;

  void setup() {
    set_buf_size();

    setup_plan(forward_plan);
    setup_plan(backward_plan);
  }

  void setup_plan(clfftPlanHandle &plan) {
    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = {nx};

    create_default_plan(plan, dim, clLengths);
    set_precision(plan, precision);
    set_inout_place(plan);
    set_data_layout(plan);
    set_batchsize(plan, M);
    
    size_t istride_t = {istride};
    size_t ostride_t = {ostride};
    set_strides(plan, dim, &istride_t, &ostride_t);

    set_dists(plan, dim, idist, odist);
    
    bake_plan(plan);
    set_workmem(plan);
  }

public:
  clmfft1() : clfft_base(), nx(0), M(0), istride(0), ostride(0), idist(0),
	      odist(0) {
    realtocomplex = false;
    set_buf_size();
  }

  clmfft1(unsigned int nx, unsigned int M, 
	  int istride, int ostride, int idist, int odist, 
	  bool inplace,
	  cl_command_queue queue, cl_context ctx) :
    clfft_base(ctx, queue, inplace, true, CLFFT_DOUBLE), 
    nx(nx), M(M), 
    istride(istride), ostride(ostride), idist(idist), odist(odist) {
    realtocomplex = false;
    setup();
  }
  
  const unsigned int ncomplex(const int dim = -1) {
    switch(dim) {
    case -1:
      return nx * M;
      break;
    case 0:
      return nx;
      break;
    default:
      std::cerr << dim
		<< " is an invalid dimension for clmfft1::ncomplex"
		<< std::endl;
      exit(1);
      return 0;
    }
    return 0;
  }

  const unsigned int nreal(const int dim = -1) {
    return 0;
  }
};

class clfft1r : public clfft_base
{
private:
  unsigned nx; // size of problem for clFFT

  void setup() {
    realtocomplex = true;
    set_buf_size();
    
    setup_plan(forward_plan, CLFFT_FORWARD);
    setup_plan(backward_plan, CLFFT_BACKWARD);
    // FIXME: delete backplan
  }

  void setup_plan(clfftPlanHandle &plan, clfftDirection direction) {
    bool forward = direction == CLFFT_FORWARD; 
 
    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = {nx};
  
    create_default_plan(plan, dim, clLengths);
    set_precision(plan, precision);
    set_data_layout(plan, forward);
    set_inout_place(plan);

    size_t iostrides[1] = {1};
    set_strides(plan, dim, iostrides, iostrides);

    size_t idist = forward ? nreal(0) : ncomplex(0);
    size_t odist = forward ? ncomplex(0) : nreal(0);
    set_dists(plan, dim, idist, odist);

    bake_plan(plan);

    set_workmem(plan);
  }

public:
  clfft1r() : clfft_base(), nx(0) {
    set_buf_size();
  }

  clfft1r(unsigned int nx, bool inplace, 
	  cl_command_queue queue, cl_context ctx) :
    clfft_base(ctx, queue, inplace, true, CLFFT_DOUBLE), nx(nx) {
    setup();
  }

  ~clfft1r() {
  }

  const unsigned int ncomplex(const int dim = -1) {
    switch(dim) {
    case -1:
      return 1 + nx / 2;
    case 0:
      return 1 + nx / 2;
    default:
      std::cerr << dim 
		<< " is an invalid dimension for clfft1r::ncomplex"
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
		<< " is an invalid dimension for clfft1r::nreal"
		<< std::endl;
      exit(1);
    }
    return 0;
  }

  size_t complex_buf_size() {
    return  (1 + nx / 2) * 2 * realsize;
  }

  size_t real_buf_size() {
    return nx * realsize;
  }

};

class clfft2r : public clfft_base
{
private:
  unsigned nx, ny; // size of problem for clFFT

  void setup() {
    realtocomplex = true;

    set_buf_size();

    setup_plan(forward_plan, CLFFT_FORWARD);
    setup_plan(backward_plan, CLFFT_BACKWARD);
    // FIXME: delete backplan
  }

  void setup_plan(clfftPlanHandle &plan, clfftDirection direction) {
    bool forward = direction == CLFFT_FORWARD; 
    clfftDim dim = CLFFT_2D;
    //size_t clLengths[2] = {nx, ny};
    size_t clLengths[2] = {ny, nx}; // They lied when they said it was row-major

    create_default_plan(plan, dim, clLengths);
    set_precision(plan, precision);
    set_data_layout(plan, forward);
    set_inout_place(plan);
    set_precision(plan, precision);

    if(forward) {
      size_t istride[2] = {1, nreal(1)};
      size_t ostride[2] = {1, ncomplex(1) + inplace};
      set_strides(plan, dim, istride, ostride);
    } else {
      size_t istride[2] = {1, ncomplex(1) + inplace};
      size_t ostride[2] = {1, nreal(1)};
      set_strides(plan, dim, istride, ostride);
    }

    size_t idist = forward ? 2 * nreal(-1) : ncomplex(-1);
    size_t odist = forward ? ncomplex(-1) : 2 * nreal(-1);
    set_dists(plan, dim, idist, odist);
    
    bake_plan(plan);
    set_workmem(plan);
  }

public:
  clfft2r() : clfft_base(), nx(0), ny(0) {
    set_buf_size();
  }

  clfft2r(unsigned int nx, unsigned int ny, bool inplace, 
	  cl_command_queue queue, cl_context ctx) :
    clfft_base(ctx, queue, inplace, true, CLFFT_DOUBLE), nx(nx), ny(ny) {
    realtocomplex = true;
    setup();
  }

  ~clfft2r() {
  }

  const unsigned int ncomplex(const int dim = -1) {
    switch(dim) {
    case -1:
      return nx * (1 + ny / 2 + inplace);
    case 0:
      return nx;
    case 1:      
      return 1 + ny / 2;
    default:
      std::cerr << dim 
		<< " is an invalid dimension for clfft2r::ncomplex"
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
		<< " is an invalid dimension for clfft2r::ncomplex"
		<< std::endl;
      exit(1);
    }
    return 0;
  }
};

class clfft3r : public clfft_base
{
private:
  unsigned nx, ny, nz; // size of problem for clFFT

  void setup() {
    realtocomplex = true;

    set_buf_size();

    setup_plan(forward_plan, CLFFT_FORWARD);
    setup_plan(backward_plan, CLFFT_BACKWARD);
    // FIXME: delete backplan
  }

  void setup_plan(clfftPlanHandle &plan, clfftDirection direction) {
    bool forward = (direction == CLFFT_FORWARD); 
    clfftDim dim = CLFFT_3D;
    size_t clLengths[3] = {nz, ny, nx};

    create_default_plan(plan, dim, clLengths);
    set_precision(plan, precision);
    set_data_layout(plan, forward);
    set_inout_place(plan);
    set_precision(plan, precision);

    // size_t istride[3] = {1, nreal(2), nreal(1) * nreal(2)};
    // size_t ostride[3] = {1, ncomplex(2), ncomplex(1) * (ncomplex(2) + inplace)};
    size_t istride[3] = {1, nreal(2), nreal(1) * nreal(2)};
    size_t ostride[3] = {1, 
			 ncomplex(2) + inplace, 
			 ncomplex(1) * (ncomplex(2) + inplace)};
    if(forward) {
      set_strides(plan, dim, istride, ostride);
    } else {
      set_strides(plan, dim, ostride, istride);
    }

    size_t idist = forward ? 2 * nreal(-1) : ncomplex(-1);
    size_t odist = forward ? ncomplex(-1) : 2 * nreal(-1);
    set_dists(plan, dim, idist, odist);
    
    bake_plan(plan);
    set_workmem(plan);
  }

public:
  clfft3r() : clfft_base(), nx(0), ny(0), nz(0) {
    set_buf_size();
  }

  clfft3r(unsigned int nx, unsigned int ny, unsigned int nz, bool inplace, 
	  cl_command_queue queue, cl_context ctx) :
    clfft_base(ctx, queue, inplace, true, CLFFT_DOUBLE), nx(nx), ny(ny), nz(nz){
    realtocomplex = true;
    setup();
  }

  ~clfft3r() {
  }

  const unsigned int ncomplex(const int dim = -1) {
    switch(dim) {
    case -1:
      return nx * ny * (1 + nz / 2 + inplace);
    case 0:
      return nx;
    case 1:
      return ny;
    case 2:
      return 1 + nz / 2;
    default:
      std::cerr << dim 
		<< " is an invalid dimension for clfft3r::ncomplex"
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
	return nx * ny * nz;
    case 0:
      return nx;
    case 1:
      return ny;
    case 2:
      return nz;
    default:
      std::cerr << dim 
		<< " is an invalid dimension for clfft3r::nreal"
		<< std::endl;
      exit(1);
    }
    return 0;
  }
};

class clmfft1r : public clfft_base
{
private:
  unsigned int nx;
  unsigned int M;
  unsigned int istride, ostride;
  unsigned int idist, odist;

  void setup() {
    realtocomplex = true;
    set_buf_size();
    
    setup_plan(forward_plan, CLFFT_FORWARD);
    setup_plan(backward_plan, CLFFT_BACKWARD);
  }

  void setup_plan(clfftPlanHandle &plan, clfftDirection direction) {
    bool forward = direction == CLFFT_FORWARD; 
    
    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = {nx};

    create_default_plan(plan, dim, clLengths);
    set_precision(plan, precision);
    if(forward)
      set_data_layout_real_to_complex(plan);
    else
      set_data_layout_complex_to_real(plan);
    set_data_layout(plan, forward);
    set_inout_place(plan);

    set_batchsize(plan, M);

    size_t istride_t = istride;
    size_t ostride_t = ostride;
    if(forward)
      set_strides(plan, dim, &istride_t, &ostride_t);
    else
      set_strides(plan, dim, &ostride_t, &istride_t);

    // FIXME: correct for in-place?
    size_t idist_t = idist;
    size_t odist_t = odist;
    if(forward)
      set_dists(plan, dim, idist_t, odist_t);
    else
      set_dists(plan, dim, odist_t, idist_t);

    if(forward)
      std::cout << "forward" << std::endl;
    else
      std::cout << "backward" << std::endl;
    std::cout << "\tistride: " << istride << std::endl;
    std::cout << "\tostride: " << ostride << std::endl;
    std::cout << "\tidist: " << idist << std::endl;
    std::cout << "\todist: " << odist << std::endl;

    bake_plan(plan);
    set_workmem(plan);
  }

public:
  clmfft1r() : clfft_base(), nx(0), M(0) {
    set_buf_size();
  }

  clmfft1r(unsigned int nx, unsigned int M, int istride, int ostride,
	   int idist, int odist,  
	   bool inplace, 
	   cl_command_queue queue, cl_context ctx) :
    clfft_base(ctx, queue, inplace, true, CLFFT_DOUBLE), 
    nx(nx), M(M), 
    istride(istride), ostride(ostride), idist(idist), odist(odist) {
    setup();
  }

  const unsigned int ncomplex(const int dim = -1) {
    switch(dim) {
    case -1:
      return (1 + nx / 2) * M;
    case 0:
      return (1 + nx / 2);
    default:
      std::cerr << dim 
		<< " is an invalid dimension for clmfft1r::ncomplex"
		<< std::endl;
      exit(1);
    }
    return 0;
  }

  const unsigned int nreal(const int dim = -1) {
    switch(dim) {
    case -1:
      if(inplace)
	return ncomplex(-1) * 2;
      else 
	return nx * M;
    case 0:
      return nx;
    default:
      std::cerr << dim 
		<< " is an invalid dimension for clmfft1r::nreal"
		<< std::endl;
      exit(1);
    }

    return 0;
  }
};
