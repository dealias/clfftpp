#ifndef __fft_hpp__
#define __fft_hpp__

#include <CL/cl.hpp>

#include <streambuf>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <clutils.h>

#include <platform.hpp>

class cl_base
{
private:
protected:
  cl_device_id device;
  cl_platform_id platform;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_mem memobj;
  unsigned int n;
  size_t maxworkgroupsize;
  std::string source_str;
  size_t size;
public:
  cl_base(){
    // FIXME: set up OpenCL environment?

    device = 0;
    platform = 0;
    context = 0;
    queue = 0;
    program = 0;
    memobj = 0;
    n = 0;
  }
  ~cl_base() {

    // FIXME:only release if created by class
    // if(queue != 0)
    //   clReleaseCommandQueue(queue);
    // if(context != 0)
    //   clReleaseContext(context);
  }
    
  void check_cl_ret(cl_int ret, const char* name)
  {
    if(ret != CL_SUCCESS) {
    
      std::cerr << "ERROR: " << name 
		<< " with retval: " << ret 
		<< " : " << clErrorString(ret) 
		<< std::endl;
    }
    assert(ret == CL_SUCCESS);
  }

  void set_device(const unsigned int nplatform, const unsigned int ndevice) {
    std::vector<std::vector<cl_device_id> > dev_ids;
    create_device_tree(dev_ids);
    device = dev_ids[nplatform][ndevice];
  }

  void set_context() {
    assert(device != 0);
    context=create_context(platform, {device});
  }

  void set_queue() {
    assert(context != 0);
    queue = create_queue(context, {device});
  }

  void alloc_rw() {
    cl_int ret;
    clCreateBuffer(context, 
		   CL_MEM_READ_WRITE ,
		   n * size, 
		   NULL, 
		   &ret);
    check_cl_ret(ret,"clCreateBuffer");
  }

  void set_maxworkgroupsize() {
    cl_int ret;
    ret = clGetDeviceInfo(device,
			  CL_DEVICE_MAX_WORK_GROUP_SIZE,
			  sizeof(maxworkgroupsize),
			  &maxworkgroupsize,
			  NULL);
    check_cl_ret(ret,"max_workgroup_size");
  }

  void read_file(const char* filename)
  {
    std::ifstream t(filename);

    t.seekg(0, std::ios::end);  
    source_str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);
  
    source_str.assign(std::istreambuf_iterator<char>(t),
		      std::istreambuf_iterator<char>());
  }

  void create_program() {
    cl_int ret;
    size_t source_size=source_str.length();
    program = clCreateProgramWithSource(context, 
					1, //number of strings passed
					(const char **)&source_str,
					(const size_t *)&source_size,
					&ret);
    check_cl_ret(ret,"clCreateProgrammWithSource"); 
  }

  void build_program() {
        cl_int ret;
    ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if(ret != CL_SUCCESS)
      std::cout <<  print_build_debug(program,&device) << std::endl;
    check_cl_ret(ret,"clBuildProgram");
  }

  void create_kernel(const char *kernelname) {
    cl_int ret;
    kernel = clCreateKernel(program, kernelname, &ret);
    check_cl_ret(ret,"create kernel");
  }

  void build_kernel_from_file(const char* filename, char* kernelname) {
    read_file(filename);
    create_program();
    build_program();
    create_kernel(kernelname);
  }

  void finish() {
    cl_int ret = clFinish(queue);
    check_cl_ret(ret,"clFinish");
  }

};

template<class T>
class mfft1d : public cl_base {
private:
  unsigned int nx, ny;
public:
  mfft1d() {
    size=sizeof(T);
  }
  mfft1d(unsigned int n0) {
    size=sizeof(T);
    n=n0;
  }
  mfft1d(unsigned int nplat, unsigned int ndev,
	 unsigned int nx0, unsigned int ny0) {
    size=sizeof(T);
    nx = nx0;
    ny = ny0;
    n=nx*ny;
    set_device(nplat,ndev);
    set_maxworkgroupsize();
    std::cout << "maxworkgroupsize: " << maxworkgroupsize << std::endl;
    set_context();
    set_queue();
  }

  mfft1d(cl_command_queue queue0, cl_context context0, cl_device_id device0,
	 unsigned int nx0, unsigned int ny0) {
    size=sizeof(T);
    nx = nx0;
    ny = ny0;
    n=nx*ny;
    device=device0;
    set_maxworkgroupsize();
    std::cout << "maxworkgroupsize: " << maxworkgroupsize << std::endl;
    queue=queue0;
    context=context0;
  }

  ~mfft1d() {
    // FIXME
  }

  void build() {
    build_kernel_from_file("mfft1.cl","mfft1");
  }
  
  void set_args(cl_mem buf=0) {
    cl_int ret;
    assert(kernel != 0);
    ret = clSetKernelArg(kernel, 0, sizeof(unsigned int),  (void *)&nx);
    check_cl_ret(ret,"setargs0");
    ret = clSetKernelArg(kernel, 1, sizeof(unsigned int),  (void *)&ny);
    check_cl_ret(ret,"setargs0");
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &(buf == 0 ? memobj : buf));
    check_cl_ret(ret,"setargs1");
  }

  void create_kernel() {
    create_kernel("mfft1d");
  }

  void write_buffer(T *f, cl_mem buf=0) {
    cl_int ret;
    ret = clEnqueueWriteBuffer(queue,
  			       buf == 0 ? memobj : buf,
  			       CL_TRUE,
  			       0,
  			       n * sizeof(T),
  			       f,
  			       0,
  			       NULL,
  			       NULL );
    check_cl_ret(ret,"clEnqueueWriteBuffer");  
    finish();
  }

  void read_buffer(T *f, cl_mem buf=0) {
    cl_int ret;
    ret = clEnqueueReadBuffer(queue,
			      buf == 0 ? memobj : buf,
  			      CL_TRUE,
  			      0,
  			      n * sizeof(T),
  			      f,
  			      0,
  			      NULL,
  			      NULL );
    check_cl_ret(ret,"clEnqueueReadBuffer");
    finish();
  }
  
  void forward() {
    cl_int ret;
    // FIXME: work sizes must change based on problem size and device
    const size_t global_work_size=nx;
    const size_t local_work_size=nx;
    ret = clEnqueueNDRangeKernel(queue,
    				 kernel,
    				 1 ,//cl_uint work_dim,
    				 NULL, //const size_t *global_work_offset,
    				 &global_work_size, //const size_t *global_work_size,
    				 &local_work_size, //const size_t *local_work_size,
    				 0, //cl_uint num_events_in_wait_list,
    				 NULL, //const cl_event *event_wait_list,
    				 NULL //cl_event *event
    				 );
    check_cl_ret(ret,"Forward");
    finish();
  }
  
  // FIXME
};

#endif
