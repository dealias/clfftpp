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

template< class T>
void init(const unsigned int nx, const unsigned int ny, T *f)
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

template< class T>
void show(const unsigned int nx, const unsigned int ny, T *f, 
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

  unsigned int outlimit=100;

  float *f=new float[2*nx*ny];

  std::cout << "Input:" << std::endl;
  init(nx,ny,f);
  show(nx,ny,f,outlimit);
    
  double *T=new double[N];
  for(unsigned int i=0; i < N; ++i) {
    // FIXME
    seconds();
    // FIXME
    T[i]=seconds();
  }
  //timings("mfft1d",nx,T,N,stats);


  //mfft1d <float>fft(platnum,devnum,nx,ny);
  mfft1d <float>fft(queue,ctx,device,nx,ny);
  fft.build();
  fft.alloc_rw();
  fft.set_args();
  init(nx,ny,f);
  fft.write_buffer(f);
  fft.forward();
  fft.finish();
  fft.read_buffer(f);
  show(nx,ny,f,outlimit);

  /* Release OpenCL working objects. */
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);
}
