/*
  OpenCL Utilities
*/

#include "clutils.h"

const char *clErrorString(const cl_int err)
{
  const char *errString = NULL;

  switch (err) {
  case CL_SUCCESS:
    errString = "CL_SUCCESS";
    break;
            
  case CL_DEVICE_NOT_FOUND:
    errString = "CL_DEVICE_NOT_FOUND";
    break;
    
  case CL_DEVICE_NOT_AVAILABLE:
    errString = "CL_DEVICE_NOT_AVAILABLE";
    break;
    
  case CL_COMPILER_NOT_AVAILABLE:
    errString = "CL_COMPILER_NOT_AVAILABLE";
    break;
            
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    errString = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    break;
    
  case CL_OUT_OF_RESOURCES:
    errString = "CL_OUT_OF_RESOURCES";
    break;
            
  case CL_OUT_OF_HOST_MEMORY:
    errString = "CL_OUT_OF_HOST_MEMORY";
    break;
            
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    errString = "CL_PROFILING_INFO_NOT_AVAILABLE";
    break;
            
  case CL_MEM_COPY_OVERLAP:
    errString = "CL_MEM_COPY_OVERLAP";
    break;
            
  case CL_IMAGE_FORMAT_MISMATCH:
    errString = "CL_IMAGE_FORMAT_MISMATCH";
    break;
            
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    errString = "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    break;
            
  case CL_BUILD_PROGRAM_FAILURE:
    errString = "CL_BUILD_PROGRAM_FAILURE";
    break;
            
  case CL_MAP_FAILURE:
    errString = "CL_MAP_FAILURE";
    break;
            
  case CL_INVALID_VALUE:
    errString = "CL_INVALID_VALUE";
    break;
            
  case CL_INVALID_DEVICE_TYPE:
    errString = "CL_INVALID_DEVICE_TYP";
    break;
            
  case CL_INVALID_PLATFORM:
    errString = "CL_INVALID_PLATFORM";
    break;
            
  case CL_INVALID_DEVICE:
    errString = "CL_INVALID_DEVICE";
    break;
            
  case CL_INVALID_CONTEXT:
    errString = "CL_INVALID_CONTEXT";
    break;
            
  case CL_INVALID_QUEUE_PROPERTIES:
    errString = "CL_INVALID_QUEUE_PROPERTIES";
    break;
            
  case CL_INVALID_COMMAND_QUEUE:
    errString = "CL_INVALID_COMMAND_QUEUE";
    break;
            
  case CL_INVALID_HOST_PTR:
    errString = "CL_INVALID_HOST_PTR";
    break;
            
  case CL_INVALID_MEM_OBJECT:
    errString = "CL_INVALID_MEM_OBJECT";
    break;
            
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    errString = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    break;
            
  case CL_INVALID_IMAGE_SIZE:
    errString = "CL_INVALID_IMAGE_SIZE";
    break;
            
  case CL_INVALID_SAMPLER:
    errString = "CL_INVALID_SAMPLER";
    break;
            
  case CL_INVALID_BINARY:
    errString = "CL_INVALID_BINARY";
    break;
            
  case CL_INVALID_BUILD_OPTIONS:
    errString = "CL_INVALID_BUILD_OPTIONS";
    break;
            
  case CL_INVALID_PROGRAM:
    errString = "CL_INVALID_PROGRAM";
    break;
            
  case CL_INVALID_PROGRAM_EXECUTABLE:
    errString = "CL_INVALID_PROGRAM_EXECUTABLE";
    break;
            
  case CL_INVALID_KERNEL_NAME:
    errString = "CL_INVALID_KERNEL_NAME";
    break;
            
  case CL_INVALID_KERNEL_DEFINITION:
    errString = "CL_INVALID_KERNEL_DEFINITION";
    break;
            
  case CL_INVALID_KERNEL:
    errString = "CL_INVALID_KERNEL";
    break;
            
  case CL_INVALID_ARG_INDEX:
    errString = "CL_INVALID_ARG_INDEX";
    break;
            
  case CL_INVALID_ARG_VALUE:
    errString = "CL_INVALID_ARG_VALUE";
    break;
            
  case CL_INVALID_ARG_SIZE:
    errString = "CL_INVALID_ARG_SIZE";
    break;
            
  case CL_INVALID_KERNEL_ARGS:
    errString = "CL_INVALID_KERNEL_ARGS";
    break;
            
  case CL_INVALID_WORK_DIMENSION:
    errString = "CL_INVALID_WORK_DIMENSION";
    break;
            
  case CL_INVALID_WORK_GROUP_SIZE:
    errString = "CL_INVALID_WORK_GROUP_SIZE";
    break;
            
  case CL_INVALID_WORK_ITEM_SIZE:
    errString = "CL_INVALID_WORK_ITEM_SIZE";
    break;
            
  case CL_INVALID_GLOBAL_OFFSET:
    errString = "CL_INVALID_GLOBAL_OFFSET";
    break;
            
  case CL_INVALID_EVENT_WAIT_LIST:
    errString = "CL_INVALID_EVENT_WAIT_LIST";
    break;
            
  case CL_INVALID_EVENT:
    errString = "CL_INVALID_EVENT";
    break;
            
  case CL_INVALID_OPERATION:
    errString = "CL_INVALID_OPERATION";
    break;
            
  case CL_INVALID_GL_OBJECT:
    errString = "CL_INVALID_GL_OBJECT";
    break;
            
  case CL_INVALID_BUFFER_SIZE:
    errString = "CL_INVALID_BUFFER_SIZE";
    break;
            
  case CL_INVALID_MIP_LEVEL:
    errString = "CL_INVALID_MIP_LEVEL";
    break;

  default:
    errString = "UNKOWN OPENCL ERROR";
  }
  
  return errString;
}

char* print_build_debug(cl_program program, cl_device_id *device) 
{
  size_t log_size;
  clGetProgramBuildInfo(program, 
			*device,
			CL_PROGRAM_BUILD_LOG, 
			0, 
			NULL, 
			&log_size);
  char *log = (char*)calloc(log_size + 1 , sizeof(char));
  clGetProgramBuildInfo(program, 
			*device, 
			CL_PROGRAM_BUILD_LOG, 
			log_size, 
			log, 
			NULL);
  return log;
}
