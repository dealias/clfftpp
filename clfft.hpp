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

  void backward(cl_mem *inbuf0, cl_mem *outbuf0,
  		cl_uint nwait = 0, cl_event *wait = NULL, 
  		cl_event *done = NULL) {
    transform(CLFFT_BACKWARD, inbuf0, outbuf0, nwait, wait, done);
  }
};

class clfft1 : public clfft_base
{
private:
  unsigned nx; // size of problem

  void setup() {
    realtocomplex = false;

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
  }

  clfft1(unsigned int nx, bool inplace, 
	 cl_command_queue queue, cl_context ctx) :
    clfft_base(ctx, queue, inplace, true, CLFFT_DOUBLE), nx(nx) {
    setup();
  }

  ~clfft1() {
  }
};

class clfft2 : public clfft_base
{
private:
  unsigned nx, ny; // size of problem

  void setup() {
    realtocomplex = false;

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
  }

  clfft2(unsigned int nx, unsigned int ny, bool inplace,
	 cl_command_queue queue, cl_context ctx) :
    clfft_base(ctx, queue, inplace, true, CLFFT_DOUBLE), nx(nx), ny(ny) {
    setup();
  }

  ~clfft2() {
  }
};

class clfft3 : public clfft_base
{
private:
  unsigned nx, ny, nz; // size of problem

  void setup() {
    realtocomplex = false;

    setup_plan(forward_plan);
    setup_plan(backward_plan);
  }

  void setup_plan(clfftPlanHandle &plan) {

    clfftDim dim = CLFFT_3D;
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
  }

  clfft3(unsigned int nx, unsigned int ny, unsigned int nz, bool inplace,
	 cl_command_queue queue, cl_context ctx) :
    clfft_base(ctx, queue, inplace, true, CLFFT_DOUBLE), nx(nx), ny(ny), nz(nz){
    setup();
  }

  ~clfft3() {
  }
};


class clmfft1 : public clfft_base
{
private: 
  unsigned int nx;
  unsigned int M;
  size_t istride, ostride;
  size_t idist, odist;

  void setup() {
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
    
    set_strides(plan, dim, &istride, &ostride);

    set_dists(plan, dim, idist, odist);
    
    bake_plan(plan);
    set_workmem(plan);
  }

public:
  clmfft1() : clfft_base(), nx(0), M(0), istride(0), ostride(0), idist(0),
	      odist(0) {
    realtocomplex = false;
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
};

class clfft1r : public clfft_base
{
private:
  unsigned nx; // size of problem for clFFT

  void setup() {
    realtocomplex = true;
    
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

    size_t nxp = nx / 2 + 1;
    size_t idist = forward ? nx : nxp;
    size_t odist = forward ? nxp : nx;
    set_dists(plan, dim, idist, odist);

    bake_plan(plan);

    set_workmem(plan);
  }

public:
  clfft1r() : clfft_base(), nx(0) {
  }

  clfft1r(unsigned int nx, bool inplace, 
	  cl_command_queue queue, cl_context ctx) :
    clfft_base(ctx, queue, inplace, true, CLFFT_DOUBLE), nx(nx) {
    setup();
  }

  ~clfft1r() {
  }
};

class clfft2r : public clfft_base
{
private:
  unsigned nx, ny; // size of problem for clFFT

  void setup() {
    realtocomplex = true;
    setup_plan(forward_plan, CLFFT_FORWARD);
    setup_plan(backward_plan, CLFFT_BACKWARD);
    // FIXME: delete backplan
  }

  void setup_plan(clfftPlanHandle &plan, clfftDirection direction) {
    bool forward = direction == CLFFT_FORWARD; 
    clfftDim dim = CLFFT_2D;
    size_t clLengths[2] = {ny, nx};

    create_default_plan(plan, dim, clLengths);
    set_precision(plan, precision);
    set_data_layout(plan, forward);
    set_inout_place(plan);
    set_precision(plan, precision);

    size_t nyp = ny / 2 + 1;

    size_t rstride[2] = {1, inplace ? 2 * nyp : ny};
    size_t cstride[2] = {1, nyp};
    if(forward)
      set_strides(plan, dim, rstride, cstride);
    else
      set_strides(plan, dim, cstride, rstride);
    
    size_t rdist = inplace ? nx * 2 * nyp : nx * ny;
    size_t cdist = nx * nyp;
    if(forward)
      set_dists(plan, dim, rdist, cdist);
    else
      set_dists(plan, dim, cdist, rdist);

    bake_plan(plan);
    set_workmem(plan);
  }

public:
  clfft2r() : clfft_base(), nx(0), ny(0) {
  }

  clfft2r(unsigned int nx, unsigned int ny, bool inplace, 
	  cl_command_queue queue, cl_context ctx) :
    clfft_base(ctx, queue, inplace, true, CLFFT_DOUBLE), nx(nx), ny(ny) {
    realtocomplex = true;
    setup();
  }

  ~clfft2r() {
  }
};

class clfft3r : public clfft_base
{
private:
  unsigned nx, ny, nz; // size of problem for clFFT

  void setup() {
    realtocomplex = true;

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

    size_t nzp = nz / 2 + 1;
    size_t rstride[3] = {1, inplace ? 2 * nzp : nz,
			 ny * (inplace ? 2 * nzp : nz)};
    size_t cstride[3] = {1, nzp, ny * nzp};
    if(forward) {
      set_strides(plan, dim, rstride, cstride);
    } else {
      set_strides(plan, dim, cstride, rstride);
    }

    size_t rdist = nx * ny * inplace ? 2 * nzp : nz;
    size_t cdist = nx * ny * nzp;
    if(forward)
      set_dists(plan, dim, rdist, cdist);
    else
      set_dists(plan, dim, cdist, rdist);
    
    bake_plan(plan);
    set_workmem(plan);
  }

public:
  clfft3r() : clfft_base(), nx(0), ny(0), nz(0) {
  }

  clfft3r(unsigned int nx, unsigned int ny, unsigned int nz, bool inplace, 
	  cl_command_queue queue, cl_context ctx) :
    clfft_base(ctx, queue, inplace, true, CLFFT_DOUBLE), nx(nx), ny(ny), nz(nz){
    realtocomplex = true;
    setup();
  }

  ~clfft3r() {
  }
};

class clmfft1r : public clfft_base
{
private:
  unsigned int nx;
  unsigned int M;
  size_t istride, ostride;
  size_t idist, odist;

  void setup() {
    realtocomplex = true;
    
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
};
