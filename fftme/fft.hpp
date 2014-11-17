#ifndef __fft_hpp__
#define __fft_hpp__

#include <CL/cl.hpp>

#include <math.h>

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
  cl_mem memobj;
  unsigned int n; // basic problem size.
  size_t maxworkgroupsize;
  size_t local_mem_size;
  size_t max_compute_units;
  size_t constant_buffer_size;  

  std::string source_str;
  size_t size;
public:
  cl_kernel kernel;

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
  }

  void set_device(const unsigned int nplatform, const unsigned int ndevice) {
    std::vector<std::vector<cl_device_id> > dev_ids;
    create_device_tree(dev_ids);
    device = dev_ids[nplatform][ndevice];
  }

  void set_context() {
    assert(device != 0);
    cl_device_id devicelist={device};
    context=create_context(platform, devicelist);
  }

  void set_queue() {
    assert(context != 0);
    cl_device_id devicelist={device};
    queue = create_queue(context, devicelist);
  }

  void alloc_rw() {
    cl_int ret;
    memobj = clCreateBuffer(context, 
			    CL_MEM_READ_WRITE ,
			    2 * n * size, // factor of two for complex 
			    NULL, 
			    &ret);
    check_cl_ret(ret,"clCreateBuffer");
    assert(ret == CL_SUCCESS);
  }
  
  cl_mem alloc_rw(size_t bufsize) {
    cl_int ret;
    cl_mem buf = clCreateBuffer(context, 
				CL_MEM_READ_WRITE ,
				bufsize, 
				NULL, 
				&ret);
    check_cl_ret(ret,"clCreateBuffer");
    assert(ret == CL_SUCCESS);
    return buf;
  }

  void set_maxworkgroupsize() {
    cl_int ret;
    ret = clGetDeviceInfo(device,
			  CL_DEVICE_MAX_WORK_GROUP_SIZE,
			  sizeof(maxworkgroupsize),
			  &maxworkgroupsize,
			  NULL);
    std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << maxworkgroupsize 
	      << std::endl;
    check_cl_ret(ret,"max_workgroup_size");
    assert(ret == CL_SUCCESS);
  }

  void get_max_compute_units() {
    cl_int ret;
    ret = clGetDeviceInfo(device,
			  CL_DEVICE_MAX_COMPUTE_UNITS,
			  sizeof(max_compute_units),
			  &max_compute_units,
			  NULL);
    std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << local_mem_size 
	      << std::endl;
    check_cl_ret(ret, "max_compute_units");
    assert(ret == CL_SUCCESS);
  }

  void get_constant_mem_size() {
    cl_int ret;
    ret = clGetDeviceInfo(device,
			  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
			  sizeof(constant_buffer_size),
			  &constant_buffer_size,
			  NULL);
    std::cout << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << constant_buffer_size 
	      << std::endl;
    check_cl_ret(ret, "constant_buffer_size");
    assert(ret == CL_SUCCESS);
  }

  void get_local_mem_size() {
    cl_int ret;
    ret = clGetDeviceInfo(device,
			  CL_DEVICE_LOCAL_MEM_SIZE,
			  sizeof(local_mem_size),
			  &local_mem_size,
			  NULL);
    std::cout << "CL_DEVICE_LOCAL_MEM_SIZE: " << local_mem_size 
	      << std::endl;
    check_cl_ret(ret, "local_mem_size");
    assert(ret == CL_SUCCESS);
  }

  void read_file(std::string &contents, const char* filename) {
    std::ifstream t(filename);
    t.seekg(0, std::ios::end);
    source_str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);
    source_str.assign(std::istreambuf_iterator<char>(t),
		      std::istreambuf_iterator<char>());
  }

  cl_program create_program(std::string source) {
    cl_int ret;
    size_t size=source.length();
    cl_program prog = clCreateProgramWithSource(context, 
						1, //number of strings passed
						(const char **)&source,
						(const size_t *)&size,
						&ret);
    check_cl_ret(ret,"clCreateProgrammWithSource"); 
    assert(ret == CL_SUCCESS);
    return prog;
  }

  void build_program(cl_program program, const char *options = NULL) {
    cl_int ret;
    ret = clBuildProgram(program, 
			 1,
			 &device, 
			 options, // Options
			 NULL, 
			 NULL);
    if(ret != CL_SUCCESS)
      std::cout <<  print_build_debug(program,&device) << std::endl;
    check_cl_ret(ret,"clBuildProgram");
    assert(ret == CL_SUCCESS);
  }

  cl_kernel create_kernel(cl_program program, const char *kernelname) {
    cl_int ret;
    cl_kernel kernel = clCreateKernel(program, kernelname, &ret);
    check_cl_ret(ret,"create kernel");
    assert(ret == CL_SUCCESS);
    return kernel;
  }

  cl_kernel build_kernel_from_file(const char* filename, char* kernelname,
			      const char *options = NULL) {
    read_file(source_str,filename);
    program = create_program(source_str);
    build_program(program, options);
    return create_kernel(program, kernelname);
  }

  void finish() {
    cl_int ret = clFinish(queue);
    check_cl_ret(ret,"clFinish");
    assert(ret == CL_SUCCESS);
  }
};

template<class T>
class mfft1d : public cl_base {
private:
  unsigned int nx, mx, ny, stride, dist;
  size_t global_work_size;
  cl_mem cl_zetas;
  T *zetas;

  cl_kernel k_mfft1d;
  cl_kernel k_mfft1d_g;

  size_t lsize;
public:
  void set_size() {
    size = sizeof(T);
  }

  mfft1d() {
    set_size();
  }

  mfft1d(unsigned int n0) {
    set_size();
    n=n0;
  }

  void set_mx() {
    get_local_mem_size();
    get_max_compute_units();
    get_constant_mem_size();
    size_t nb = local_mem_size / (sizeof(T) * 2 * ny);
    nb = std::min(nb, maxworkgroupsize);
    mx = (nx + nb -1) / nb;
    global_work_size = (nx+mx-1)/mx;
    std::cout << "nb: " << nb << std::endl;
    lsize = sizeof(T) * 2 * nb * ny;
  }

  void create_cl_zetas() {
    cl_int ret;
    cl_zetas = clCreateBuffer(context,
			      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
			      sizeof(T) * 2 * ny,
			      zetas,
			      &ret);
    check_cl_ret(ret,"create clzetas");
    assert(ret == CL_SUCCESS);
  }

  void compute_zetas() {
    const T PI = 4.0 * atan(1.0);
    for(unsigned int i = 0; i < ny; ++i) {
      const T arg = -2.0 * PI * i / (T) ny;
      zetas[2 * i] = cos(arg);
      zetas[2 * i + 1] = sin(arg);
    }
  }

  mfft1d(cl_command_queue queue0, cl_context context0, cl_device_id device0,
	 unsigned int nx0, unsigned int ny0,
	 unsigned int stride0, unsigned int dist0) {
    set_size();
    nx = nx0;
    ny = ny0;
    n = nx * ny;
    device = device0;
    set_maxworkgroupsize();
    set_mx();
    queue = queue0;
    context = context0;
    stride = stride0;
    dist = dist0;

    zetas = new T[2 * ny];
    compute_zetas();
    create_cl_zetas();
  }

  ~mfft1d() {
    // FIXME: free cl_zetas
    delete[] zetas;
  }
  
  void build() {
    char filename[] = "mfft1.cl";
    if(std::is_same<T, double>::value) {
      {
	char kernelname[] = "mfft1";
	k_mfft1d = build_kernel_from_file(filename, kernelname,"-I double/");
      }
      {
	char kernelname[] = "mfft1_g";
	k_mfft1d_g = build_kernel_from_file(filename, kernelname,"-I double/");
      }
    }
    if(std::is_same<T, float>::value) {
      {
	char kernelname[] = "mfft1";
	k_mfft1d = build_kernel_from_file(filename, kernelname,"-I float/");

      }
      {
	char kernelname[] = "mfft1_g";
	k_mfft1d_g = build_kernel_from_file(filename, kernelname,"-I float/");
      }
    }
    
  }
  
  void set_args_mfft1d(cl_mem buf) {
    cl_int ret;
    assert(k_mfft1d != 0);
    unsigned int narg=0;

    ret = clSetKernelArg(k_mfft1d, narg++, sizeof(nx), (void *)&nx);
    check_cl_ret(ret,"setargs nx");
    assert(ret == CL_SUCCESS);

    ret = clSetKernelArg(k_mfft1d, narg++, sizeof(mx), (void *)&mx);
    check_cl_ret(ret,"setargs mx");
    assert(ret == CL_SUCCESS);

    ret = clSetKernelArg(k_mfft1d, narg++, sizeof(ny), (void *)&ny);
    check_cl_ret(ret,"setargs ny");
    assert(ret == CL_SUCCESS);

    ret = clSetKernelArg(k_mfft1d, narg++, sizeof(stride), (void *)&stride);
    check_cl_ret(ret,"setargs stride");
    assert(ret == CL_SUCCESS);

    ret = clSetKernelArg(k_mfft1d, narg++, sizeof(dist), (void *)&dist);
    check_cl_ret(ret,"setargs dist");
    assert(ret == CL_SUCCESS);

    ret = clSetKernelArg(k_mfft1d, narg++,
			 sizeof(cl_mem), &(buf == 0 ? memobj : buf));
    check_cl_ret(ret,"setargs buf");
    assert(ret == CL_SUCCESS);

    { // local memory for FFTs:
      ret = clSetKernelArg(k_mfft1d, 
      			   narg++,
      			   lsize,
      			   NULL // passing NULL allocates local memory
      			   );
      check_cl_ret(ret,"setargs local work");
    }

    { // the zetas are here!
      // Contant-memory version:
      ret = clSetKernelArg(k_mfft1d,
			   narg++,
			   sizeof(cl_mem),   
			   &cl_zetas);
      check_cl_ret(ret,"setargs zeta buf");
      assert(ret == CL_SUCCESS);

      // Local-memory version:
      // ret = clSetKernelArg(k_mfft1d, narg++,
      // 			 sizeof(cl_mem), &zbuf);
      // check_cl_ret(ret,"setargs twiddle buf");
      // assert(ret == CL_SUCCESS);
    }
  }

  void set_args_mfft1d_g(cl_mem buf) {
    cl_int ret;
    assert(k_mfft1d_g != 0);
    unsigned int narg=0;

    ret = clSetKernelArg(k_mfft1d_g, narg++, sizeof(nx), (void *)&nx);
    check_cl_ret(ret,"setargs nx");
    assert(ret == CL_SUCCESS);

    ret = clSetKernelArg(k_mfft1d_g, narg++, sizeof(mx), (void *)&mx);
    check_cl_ret(ret,"setargs mx");
    assert(ret == CL_SUCCESS);

    ret = clSetKernelArg(k_mfft1d_g, narg++, sizeof(ny), (void *)&ny);
    check_cl_ret(ret,"setargs ny");
    assert(ret == CL_SUCCESS);

    ret = clSetKernelArg(k_mfft1d_g, narg++, sizeof(stride), (void *)&stride);
    check_cl_ret(ret,"setargs stride");
    assert(ret == CL_SUCCESS);

    ret = clSetKernelArg(k_mfft1d_g, narg++, sizeof(dist), (void *)&dist);
    check_cl_ret(ret,"setargs dist");
    assert(ret == CL_SUCCESS);

    ret = clSetKernelArg(k_mfft1d_g, narg++,
			 sizeof(cl_mem), &(buf == 0 ? memobj : buf));
    check_cl_ret(ret,"setargs buf");
    assert(ret == CL_SUCCESS);

    { // local memory for FFTs:
      // ret = clSetKernelArg(k_mfft1d_g, 
      // 			   narg++,
      // 			   lsize,
      // 			   NULL // passing NULL allocates local memory
      // 			   );
      // check_cl_ret(ret,"setargs local work");
    }

    { // the zetas are here!
      // Contant-memory version:
      ret = clSetKernelArg(k_mfft1d_g,
			   narg++,
			   sizeof(cl_mem),   
			   &cl_zetas);
      check_cl_ret(ret,"setargs zeta buf");
      assert(ret == CL_SUCCESS);

      // Local-memory version:
      // ret = clSetKernelArg(k_mfft1d, narg++,
      // 			 sizeof(cl_mem), &zbuf);
      // check_cl_ret(ret,"setargs twiddle buf");
      // assert(ret == CL_SUCCESS);
    }
  }

  void set_args(cl_mem buf=0) {
    set_args_mfft1d(buf);
    set_args_mfft1d_g(buf);
  }

  void write_buffer(T *f, cl_mem buf=0) {
    cl_int ret;
    ret = clEnqueueWriteBuffer(queue,
  			       buf == 0 ? memobj : buf,
  			       CL_TRUE,
  			       0,
  			       2 * n * sizeof(T),
  			       f,
  			       0,
  			       NULL,
  			       NULL );
    check_cl_ret(ret,"clEnqueueWriteBuffer");  
    assert(ret == CL_SUCCESS);
    finish();
  }

  void read_buffer(T *f, cl_mem buf=0) {
    cl_int ret;
    ret = clEnqueueReadBuffer(queue,
			      buf == 0 ? memobj : buf,
  			      CL_TRUE, // cl_bool blocking_read,
  			      0, // offset
  			      2 * n * sizeof(T), //size_t cb
  			      f,
  			      0,
  			      NULL,
  			      NULL );
    check_cl_ret(ret,"clEnqueueReadBuffer");
    assert(ret == CL_SUCCESS);
    finish();
  }
  
  inline void forward(cl_event *event=NULL) {
    cl_int ret;
    ret = clEnqueueNDRangeKernel(queue,
    				 k_mfft1d,
    				 1 ,//cl_uint work_dim,
    				 NULL, //const size_t *global_work_offset,
    				 &global_work_size, //const size_t *global_work_size,
    				 NULL, //&local_work_size, //const size_t *local_work_size,
    				 0, //cl_uint num_events_in_wait_list,
    				 NULL, //const cl_event *event_wait_list,
    				 event//NULL //cl_event *event
    				 );
    check_cl_ret(ret,"Forward");
    assert(ret == CL_SUCCESS);
  }
  inline void forward_g(cl_event *event=NULL) {
    cl_int ret;

    size_t gwork = std::min((size_t)nx,global_work_size);
    ret = clEnqueueNDRangeKernel(queue,
    				 k_mfft1d_g,
    				 1 ,//cl_uint work_dim,
    				 NULL, //const size_t *global_work_offset,
    				 &gwork, //const size_t *global_work_size,
    				 NULL, //&local_work_size, //const size_t *local_work_size,
    				 0, //cl_uint num_events_in_wait_list,
    				 NULL, //const cl_event *event_wait_list,
    				 event//NULL //cl_event *event
    				 );
    check_cl_ret(ret,"Forward");
    assert(ret == CL_SUCCESS);
  }
};

#endif
