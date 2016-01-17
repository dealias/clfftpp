#ifndef __platform_hpp__
#define __platform_hpp__
#include <string>
#include <vector>
#include <assert.h>
extern "C" {
#include "clutils.h"
}
#include <CL/cl.h>

void show_devices();
void create_device_tree(std::vector<std::vector<cl_device_id> > &D);
void find_platform_ids(std::vector<cl_platform_id > &A);
cl_context create_context(const cl_platform_id platform,
			  const cl_device_id device);
cl_command_queue create_queue(const cl_context ctx,
			      const cl_device_id device,
			      cl_command_queue_properties properties=0);

cl_program create_program(const std::string source, cl_context context);
void build_program(cl_program program, cl_device_id device,
		   const char *options = NULL);
cl_kernel create_kernel(cl_program program, const char *kernelname);

#endif
