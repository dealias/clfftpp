/* No need to explicitely include the OpenCL headers */
#include <clFFT.h>
#include <iostream>

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
    precision=CLFFT_DOUBLE;
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
    if(err > 0) 
      std::cerr << "clfft::clfft_setup error "<< err << std::endl;
  }

  double * create_rambuf() {
    return new double[buf_size];
  }
 
  cl_mem create_clbuf() {
    cl_int err;
    bufX = clCreateBuffer(ctx, 
			  CL_MEM_READ_WRITE, 
			  buf_size,
			  NULL,
			  &err);
    if(err > 0) 
      std::cerr << "clfft::create_clbuf error "<< err << std::endl;
    return bufX;
  }

  void cl_to_ram(double *X, cl_mem bufX0=NULL) {
    cl_mem buf = (bufX0 != NULL) ? bufX0 : bufX;
    cl_int err;
    err = clEnqueueReadBuffer(queue,
			      buf,
			      CL_TRUE,
			      0,
			      buf_size,
			      X,
			      0,
			      NULL,
			      NULL );
    if (err > 0)
      std::cerr << "Error in clfft_base::cl_to_ram: " << std::endl;
  }

  void ram_to_cl(double *X, cl_mem bufX0=NULL) {
    cl_mem buf = (bufX0 != NULL) ? bufX0 : bufX;
    cl_int err;
    err = clEnqueueWriteBuffer(queue,
			       buf,
			       CL_TRUE,
			       0,
			       buf_size,
			       X,
			       0,
			       NULL,
			       NULL);
    if (err > 0)
      std::cerr << "Error in clfft_base::ram_to_cl: " << std::endl;
  }

  void wait() {
    cl_int err = clFinish(queue);
    if (err > 0)
      std::cerr << "Error in clfft_base::wait: " << err << std::endl;
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

    cl_int err;
    err = clfftCreateDefaultPlan(&plan, 
				 ctx, 
				 dim, 
				 clLengths);
    err = clfftSetPlanPrecision(plan, 
				precision);
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

    if(err > 0) 
      std::cerr << "clfft1::setup error "<< err << std::endl;
    
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
    cl_int err;
    err = clfftDestroyPlan(&plan);
    if(err > 0) 
      std::cerr << "clfft::~clfft1 error "<< err << std::endl;

  }

  void forward(cl_mem bufX0=NULL) {
    cl_mem buf = (bufX0 != NULL) ? bufX0 : bufX;
    cl_int err;
    err = clfftEnqueueTransform(plan,
				CLFFT_FORWARD,
				1,
				&queue,
				0,
				NULL,
				NULL,
				&buf,
				NULL,
				NULL);
    if (err > 0)
      std::cerr << "Error in clfft1::forward: " << err << std::endl;
  }

  void backward(cl_mem bufX0=NULL) {
    cl_mem buf = (bufX0 != NULL) ? bufX0 : bufX;
    cl_int err;
    err = clfftEnqueueTransform(plan,
				CLFFT_BACKWARD,
				1,
				&queue,
				0,
				NULL,
				NULL,
				&buf,
				NULL,
				NULL);
    if (err > 0)
      std::cerr << "Error in clfft1::backward: " << err << std::endl;
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

    cl_int err;
    err = clfftCreateDefaultPlan(&plan, 
				 ctx, 
				 dim, 
				 clLengths);
    err = clfftSetPlanPrecision(plan, 
				precision);
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

    if(err > 0) 
      std::cerr << "clfft2::setup error "<< err << std::endl;
    
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
    cl_int err;
    err = clfftDestroyPlan(&plan);
    if(err > 0) 
      std::cerr << "clfft::~clfft2 error "<< err << std::endl;
  }

  void forward(cl_mem bufX0=NULL) {
    cl_mem buf = (bufX0 != NULL) ? bufX0 : bufX;
    cl_int err;
    err = clfftEnqueueTransform(plan,
    				CLFFT_FORWARD,
    				1,
    				&queue,
    				0,
    				NULL,
    				NULL,
    				&buf,
    				NULL,
    				NULL);
    if (err > 0)
      std::cerr << "Error in clfft2::backward: " << err << std::endl;
  }

  void backward(cl_mem bufX0=NULL) {
    cl_mem buf = (bufX0 != NULL) ? bufX0 : bufX;
    cl_int err;
    err = clfftEnqueueTransform(plan,
    				CLFFT_BACKWARD,
    				1,
    				&queue,
    				0,
    				NULL,
    				NULL,
    				&buf,
    				NULL,
    				NULL);
    if (err > 0)
      std::cerr << "Error in clfft2::backward: " << err << std::endl;
  }

};


class clfft1r : public clfft_base
{
private:
  unsigned nx; // size of problem

  void set_buf_size() {
    buf_size = (nx + 1) * 2 * sizeof(double); // TODO: variable precision
  }

  void setup() {
    set_buf_size();

    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = {nx};

    cl_int err;
    err = clfftCreateDefaultPlan(&plan, 
				 ctx, 
				 dim, 
				 clLengths);
    err = clfftSetPlanPrecision(plan, 
				precision);
    err = clfftSetLayout(plan, 
			 CLFFT_COMPLEX_INTERLEAVED, 
			 CLFFT_REAL);
    err = clfftSetResultLocation(plan, 
				 CLFFT_INPLACE);
    err = clfftBakePlan(plan,
			1, // numQueues: number of experiments 
			&queue, // commQueueFFT
			NULL, // Always NULL
			NULL // Always NULL
			);

    if(err > 0) 
      std::cerr << "clfft1r::setup error "<< err << std::endl;
    
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
    cl_int err;
    err = clfftDestroyPlan(&plan);
    if(err > 0) 
      std::cerr << "clfft::~clfft1r error "<< err << std::endl;

  }

  void forward(cl_mem bufX0=NULL) {
    cl_mem buf = (bufX0 != NULL) ? bufX0 : bufX;
    cl_int err;
    err = clfftEnqueueTransform(plan,
				CLFFT_FORWARD,
				1,
				&queue,
				0,
				NULL,
				NULL,
				&buf,
				NULL,
				NULL);
    if (err > 0)
      std::cerr << "Error in clfft1r::forward: " << err << std::endl;
  }

  void backward(cl_mem bufX0=NULL) {
    cl_mem buf = (bufX0 != NULL) ? bufX0 : bufX;
    cl_int err;
    err = clfftEnqueueTransform(plan,
				CLFFT_BACKWARD,
				1,
				&queue,
				0,
				NULL,
				NULL,
				&buf,
				NULL,
				NULL);
    if (err > 0)
      std::cerr << "Error in clfft1r::backward: " << err << std::endl;
  }

};
