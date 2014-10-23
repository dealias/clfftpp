#include <vector>
#include <iostream>
//#include <stdio.h>
#include <CL/cl.h>
//using namespace opencl;

void find_platform_ids(std::vector<cl_platform_id > &A)
{
  const cl_uint max_plats=100;
  cl_uint ret_num_platforms;
  cl_platform_id platform_id[max_plats];
  clGetPlatformIDs(max_plats, platform_id, &ret_num_platforms);
  
  for(int i=0; i < ret_num_platforms; ++i) 
    A.push_back(platform_id[i]);
}

void find_device_ids(const cl_platform_id plat_id, 
		     std::vector<cl_device_id > &A)
{
  cl_int ret;
  const cl_uint max_dev=100;
  cl_device_id device_ids[max_dev];

  cl_uint ret_num_devices;
  // Obtain the list of devices available on a platform.
  ret = clGetDeviceIDs(plat_id,
		       CL_DEVICE_TYPE_ALL, // FIXME: to be set by user 
		       max_dev,  // num_entries
		       device_ids,
		       &ret_num_devices);
  for(int i=0; i < ret_num_devices; ++i) 
     A.push_back(device_ids[i]);
}

void create_device_tree(std::vector<std::vector<cl_device_id> > &D)
{
  std::vector<cl_platform_id > P;
  find_platform_ids(P);

  D.resize(P.size());
  for(int i=0; i < P.size(); ++i) {
    std::vector<cl_device_id> temp;
    find_device_ids(P[i],temp);
    D[i].resize(temp.size());
    for(int j=0; j < D[i].size(); ++j)
      D[i][j] = temp[j];
  }
}

void show_devices()
{
  std::vector<cl_platform_id > P;
  find_platform_ids(P);
  std::vector<std::vector<cl_device_id> > D;
  create_device_tree(D);  
    
  char buffer[1024];
  for(int i=0; i < P.size(); ++i) {
    cl_uint ret = clGetPlatformInfo(P[i],
			    CL_PLATFORM_NAME, //cl_device_info param_name,
			    sizeof(buffer), buffer, NULL);
    std::cout << "platform "<< i <<": " << buffer << std::endl;
    
    for(int j=0; j < D[i].size(); ++j) {
      std::cout << D[i].size() << std::endl;
      
      ret = clGetDeviceInfo(D[i][0],
			    CL_DEVICE_NAME, //cl_device_info param_name,
			    sizeof(buffer), 
			    buffer, 
			    NULL);
      std::cout << "\tdevice " << j << ": " << buffer << std::endl;
      
     }
  
  }
}

