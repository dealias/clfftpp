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
  
  for(unsigned int i=0; i < ret_num_platforms; ++i) 
    A.push_back(platform_id[i]);
}

void find_device_ids(const cl_platform_id plat_id, 
		     std::vector<cl_device_id > &A)
{
  const cl_uint max_dev=100;
  cl_device_id device_ids[max_dev];

  cl_uint ret_num_devices;
  // Obtain the list of devices available on a platform.
  cl_int ret = clGetDeviceIDs(plat_id,
		       CL_DEVICE_TYPE_ALL, // FIXME: to be set by user 
		       max_dev,  // num_entries
		       device_ids,
		       &ret_num_devices);
  if(ret > 0) 
    std::cerr << "Error in find_device_ids: " << ret << std::endl;
  for(unsigned int i=0; i < ret_num_devices; ++i) 
     A.push_back(device_ids[i]);
}

void create_device_tree(std::vector<std::vector<cl_device_id> > &D)
{
  std::vector<cl_platform_id > P;
  find_platform_ids(P);

  D.resize(P.size());
  for(unsigned int i=0; i < P.size(); ++i) {
    std::vector<cl_device_id> temp;
    find_device_ids(P[i],temp);
    D[i].resize(temp.size());
    for(unsigned int j=0; j < D[i].size(); ++j)
      D[i][j] = temp[j];
  }
}

void show_devices()
{
  std::vector<cl_platform_id > P;
  find_platform_ids(P);
  std::vector<std::vector<cl_device_id> > D;
  create_device_tree(D);  
  cl_uint ret;

  char buffer[1024];
  for(unsigned int i=0; i < P.size(); ++i) {
    ret = clGetPlatformInfo(P[i],
			    CL_PLATFORM_NAME, //cl_device_info param_name,
			    sizeof(buffer), buffer, NULL);
    std::cout << "platform "<< i <<": " << buffer << std::endl;
    
    for(unsigned int j=0; j < D[i].size(); ++j) {
      /*ret = clGetDeviceInfo(D[i][0],
			    CL_DEVICE_NAME, //cl_device_info param_name,
			    sizeof(buffer), 
			    buffer, 
			    NULL);*/
      ret = clGetDeviceInfo(D[i][j],
                            CL_DEVICE_NAME, //cl_device_info param_name,
                            sizeof(buffer), 
                            buffer, 
                            NULL);
      std::cout << "\tdevice " << j << " name: " << buffer << std::endl;

      cl_device_type dtype;
      ret = clGetDeviceInfo(D[i][j],
                            CL_DEVICE_TYPE, //cl_device_info param_name,
                            sizeof(dtype), 
                            &dtype, 
                            NULL);
      std::cout << "\t         type: " << dtype << std::endl;
     }
  }
  if(ret > 0) 
    std::cerr << "Error in find_device_ids: " << ret << std::endl;

}

cl_context create_context(const cl_platform_id platform,
			  const cl_device_id device)
{
  cl_int err;
  cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, 0, 0};
  props[1] = (cl_context_properties) platform;
  return clCreateContext(props, 1, &device, NULL, NULL, &err);
}

cl_command_queue create_queue(const cl_context ctx, 
			      const cl_device_id device,
			      cl_command_queue_properties properties)
{
  cl_int err;
  return clCreateCommandQueue(ctx, 
			      device, 
			      properties,
			      &err);
}
