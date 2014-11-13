#ifndef __platform_hpp__
#define __platform_hpp__

#include <vector>

#include <CL/cl.h>

//cl_device_id
void show_devices();
void create_device_tree(std::vector<std::vector<cl_device_id> > &D);
void find_platform_ids(std::vector<cl_platform_id > &A);
cl_context create_context(const cl_platform_id platform,
			  const cl_device_id device);
cl_command_queue create_queue(const cl_context ctx,
			      const cl_device_id device,
			      cl_command_queue_properties properties=0);
#endif
