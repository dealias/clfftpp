#include <string>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <platform.hpp>

#include <clutils.h>
#include <timing.h>
#include <seconds.h>
#include <assert.h>

#include <fft.hpp>

#include <getopt.h>
#include <CL/cl.hpp>

void read_file(std::string &str, const char* filename)
{
  std::ifstream t(filename);

  t.seekg(0, std::ios::end);  
  str.reserve(t.tellg());
  t.seekg(0, std::ios::beg);
  
  str.assign(std::istreambuf_iterator<char>(t),
	     std::istreambuf_iterator<char>());
}

void check_cl_ret(cl_int ret, const char* msg)
{
  if(ret != CL_SUCCESS) {
    
    std::cerr << "ERROR: " << msg 
	      << " with retval: " << ret 
	      << " : " << clErrorString(ret) 
	      << std::endl;
  }
  assert(ret == CL_SUCCESS);
}

void init(const unsigned int nx, const unsigned int ny, float*f)
{
  for(unsigned int ix=0; ix < nx; ++ix) {
    unsigned int iy;
    for(iy=0; iy < ny; ++iy) {
      unsigned int pos=2*(ix*ny + iy);
      f[pos]=ix;
      f[pos+1]=iy; /* iy+1; */
    }
  }
}

void show(const unsigned int nx, const unsigned int ny, float*f, 
	  unsigned int outlimit)
{

  if(nx*ny < outlimit) {
    for(unsigned int ix=0; ix < nx; ++ix) {
      for(unsigned int iy=0; iy < ny; ++iy) {
	int pos=2*(ix*ny + iy); 
	std::cout << "(" << f[pos] << "," << f[pos+1] << ") ";
      }
      std::cout << std::endl;
    }
  } else {
    std::cout << "(" << f[0] << "," << f[1] << ") " << std::endl;
  }
}


int main(int argc, char* argv[])
{
  // Set up the OpenCL device, platform, queue, and context.
  show_devices();

  int platnum=0;
  int devnum=0;
  
  unsigned int nx = 4;
  unsigned int ny = 4;
  //nx=262144;

  unsigned int N=10;

  unsigned int stats=MEAN; // Type of statistics used in timing test.

  for (;;) {
    int c = getopt(argc,argv,"p:d:m:x:y:N:S:h");
    if (c == -1) break;
    
    switch (c) {
    case 'p':
      platnum=atoi(optarg);
      break;
    case 'd':
      devnum=atoi(optarg);
      break;
    case 'x':
      nx=atoi(optarg);
      break;
    case 'y':
      ny=atoi(optarg);
      break;
    case 'm':
      nx=atoi(optarg);
      ny=atoi(optarg);
      break;
    case 'N':
      N=atoi(optarg);
      break;
    case 'S':
      nx=atoi(optarg);
      break;
    case 'h':
      //usage(1);
      exit(0);
      break;
    default:
      std::cout << "Invalid option" << std::endl;
      //usage(1);
      exit(1);
    }
  }

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
  read_file(source_str,"mfft1.cl");
  //std::cout << source_str << std::endl;

  cl_int ret; // return values from the OpenCL operations.

  size_t source_size=source_str.length();
  // Create Kernel Program from the source
  cl_program program 
    = clCreateProgramWithSource(ctx, 
				1, 
				(const char **)&source_str,
				(const size_t *)&source_size,
				&ret);
  check_cl_ret(ret,"clCreateProgrammWithSource"); 

  // Build Kernel Program
  ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if(ret != CL_SUCCESS)
    std::cout <<  print_build_debug(program,&device) << std::endl;
  check_cl_ret(ret,"clBuildProgram");

  // Create OpenCL Kernel
  cl_kernel kernel = clCreateKernel(program, "mfft1", &ret);
  check_cl_ret(ret,"create kernel");

  float *f=new float[2*nx*ny];

  cl_mem memobj = clCreateBuffer(ctx, 
				 CL_MEM_READ_WRITE ,
				 2 * nx * ny * sizeof(float), 
				 NULL, 
				 &ret);

  // Set OpenCL Kernel Parameters

  ret = clSetKernelArg(kernel, 0, sizeof(unsigned int),  (void *)&nx);
  check_cl_ret(ret,"setargs0");
  ret = clSetKernelArg(kernel, 1, sizeof(unsigned int),  (void *)&ny);
  check_cl_ret(ret,"setargs0");
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memobj);
  check_cl_ret(ret,"setargs1");
  
  // // Execute OpenCL Kernel
  // ret = clEnqueueTask(queue, 
  // 		      kernel, 
  // 		      0,
  // 		      NULL,
  // 		      NULL);
  // check_cl_ret(ret,"clEnqueueTask");

  const size_t global_work_size=nx;
  const size_t local_work_size=nx;

  unsigned int outlimit=100;

  std::cout << "Input:" << std::endl;
  init(nx,ny,f);
  show(nx,ny,f,outlimit);
    
  double *T=new double[N];
  for(unsigned int i=0; i < N; ++i) {
    init(nx,ny,f);
    ret = clEnqueueWriteBuffer(queue,
			       memobj,
			       CL_TRUE,
			       0,
			       2 * nx * ny * sizeof(float),
			       f,
			       0,
			       NULL,
			       NULL );
    check_cl_ret(ret,"clEnqueueWriteBuffer");  
    

    seconds();

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

    T[i]=seconds();
  }
  check_cl_ret(ret,"clEnqueueNDRangeKernel");
  timings("mfft1d",nx,T,N,stats);

  ret = clEnqueueReadBuffer(queue,
			    memobj,
			    CL_TRUE,
			    0,
			    2 * nx * ny * sizeof(float),
			    f,
			    0,
			    NULL,
			    NULL );
  check_cl_ret(ret,"clEnqueueReadBuffer");  

  ret = clFinish(queue);

  std::cout << "Output:" << std::endl;
  show(nx,ny,f,outlimit);

  check_cl_ret(ret,"clFinish");

  std::cout << "With class:" << std::endl;

  //mfft1d <float>fft(platnum,devnum,nx,ny);
  mfft1d <float>fft(queue,ctx,device,nx,ny);
  fft.build();
  fft.set_args(memobj);
  //fft.alloc_rw();
  init(nx,ny,f);
  fft.write_buffer(f,memobj);
  fft.forward();
  fft.read_buffer(f,memobj);
  show(nx,ny,f,outlimit);

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);
}
