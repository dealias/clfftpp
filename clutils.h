/*
  OpenCL Utility functions
*/

#ifndef CLUTILS_H_INC
#define CLUTILS_H_INC

#include <CL/cl.h>

const char* clErrorString(const cl_int err);

char* print_build_debug(cl_program program, cl_device_id *device);
#endif /* CLUTILS_H_INC */

