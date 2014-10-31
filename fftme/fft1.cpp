#include <string>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <platform.hpp>

#include <CL/cl.hpp>

void read_file(std::string &str, const char* filename)
{
  std::ifstream t(filename);

  t.seekg(0, std::ios::end);  
  str.reserve(t.tellg());
  t.seekg(0, std::ios::beg);
  
  str.assign((std::istreambuf_iterator<char>(t)),
	     std::istreambuf_iterator<char>());
}

void check_cl_ret(cl_int ret, const char* msg) 
{
  if(ret != CL_SUCCESS) {
    std::cout << "ERROR: " << msg << " with retval: " << ret << std::endl;
  }
}

int main()
{

  // Set up the OpenCL device, platform, queue, and context.
  show_devices();

  int platnum=0;
  int devnum=0;
  
  std::vector<std::vector<cl_device_id> > dev_ids;
  create_device_tree(dev_ids);
  cl_device_id device = dev_ids[platnum][devnum];

  std::vector<cl_platform_id> plat_ids;
  find_platform_ids(plat_ids);
  cl_platform_id platform = plat_ids[devnum];

  const cl_context ctx = create_context(platform, {device});
  const cl_command_queue queue = create_queue(ctx, {device});

  // Read the kernel from a file.
  std::string source_str;
  read_file(source_str,"fft1cc.cl");
  std::cout << source_str << std::endl;

  cl_int ret; // return valoes from the OpenCL operations.

  size_t source_size=source_str.length();
  // Create Kernel Program from the source
  cl_program program = clCreateProgramWithSource(ctx, 
						 1, 
						 (const char **)&source_str,
						 (const size_t *)&source_size,
						 &ret);
  check_cl_ret(ret,"clCreateProgrammWithSource");  

  // Build Kernel Program
  ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  check_cl_ret(ret,"clBuildProgram");

  // Create OpenCL Kernel
  cl_kernel kernel = clCreateKernel(program, "fft1cc", &ret);
  check_cl_ret(ret,"create kernel");

  int n=8;

  float *f=new float[2*n];

  cl_mem memobj = clCreateBuffer(ctx, 
				 CL_MEM_READ_WRITE ,
				 2 * n * sizeof(float), 
				 NULL, 
				 &ret);

  // Set OpenCL Kernel Parameters

  ret = clSetKernelArg(kernel, 0, sizeof(unsigned int),  (void *)&n);
  check_cl_ret(ret,"setargs0");
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memobj);
  check_cl_ret(ret,"setargs1");
  
  // Execute OpenCL Kernel
  ret = clEnqueueTask(queue, 
		      kernel, 
		      0,
		      NULL,
		      NULL);
  check_cl_ret(ret,"clEnqueueTask");

  ret = clEnqueueReadBuffer(queue,
			    memobj,
			    CL_TRUE,
			    0,
			    2* n * sizeof(float),
			    f,
			    0,
			    NULL,
			    NULL );
  check_cl_ret(ret,"clEnqueueReadBuffer");  

  ret = clFinish(queue);
  check_cl_ret(ret,"clFinish");

  std::cout << f[1] << std::endl;
  

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);
}
