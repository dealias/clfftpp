/*
This file is part of clFFT++.

clFFT++ is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

clFFT++ is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with clFFT++.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __platform_hpp__
#define __platform_hpp__
#include <string>
#include <vector>
#include <assert.h>
extern "C" {
#include "clutils.h"
}
#include <CL/cl.h>

namespace platform{
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
}
#endif
