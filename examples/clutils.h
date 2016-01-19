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

#ifndef CLUTILS_H_INC
#define CLUTILS_H_INC

#include <stdlib.h>
#include <CL/cl.h>

const char *clErrorString(const cl_int err);
char *print_build_debug(cl_program program, cl_device_id *device);

#endif /* CLUTILS_H_INC */
