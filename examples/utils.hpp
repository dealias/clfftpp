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

#ifndef __utils_hpp__
#define __utils_hpp__ 1

#include <iostream>

void usage(unsigned int dim, bool mfft = false)
{
  std::cout << "usage:\n"
	    << "./clfft1 \n"
	    << "\t-P <int>\tOpenCL platform number \n"
	    << "\t-D <int>\tOpenCL device number \n"
	    << "\t-c <0 or 1>\tinclude memory copy in time \n"
	    << "\t-m <int>\tproblem size \n"
	    << "\t-x <int>\tproblem size in first dimension \n"
	    << "\t-i <bool>\tin-place (1) or out-of-place (0) transform \n";
  if(dim > 1)
    std::cout <<  "\t-y <int>\tproblem size in second dimension\n";
  if(dim > 2)
    std::cout <<  "\t-z <int>\tproblem size in third dimension\n";
  std::cout << "\t-N <int>\tNumber of tests\n"
	    << "\t-S <int>\tStatistical measure to use\n";
  if(mfft) {
    std::cout << "\t-M <int>\tNumber of transforms\n"
	      << "\t-n <int>\tLength of transforms\n"
	      << "\t-s <int>\tInput stride\n"
	      << "\t-t <int>\tOutput stride\n"
	      << "\t-d <int>\tInput distance\n"
	      << "\t-e <int>\tOutput distance\n";
  }
  std::cout << std::endl;
}

template<class T>
void show1C(const T *X, const unsigned int n)
{
  for(unsigned int i = 0; i < n; ++i) {
    std::cout << "(" 
	      << X[2 * i] 
	      << "," 
	      <<  X[2 * i +1] 
	      << ") "; 
  }
  std::cout << std::endl;
}

template<class T>
void show1C(const T *X, const unsigned int nx, const unsigned int M, 
	    const unsigned int dist = 0)
{
  const unsigned int dist0 = dist == 0 ? nx : dist;
  for(unsigned int m = 0; m < M; ++m) {
    for(unsigned int i = 0; i < nx; ++i) {
      int pos = dist0 * m + i;
      std::cout << "(" 
		<< X[2 * pos]
		<< ","
		<<  X[2 * pos +1] 
		<< ") ";
    }
    std::cout << std::endl;
  }
}

template<class T>
void show2C(const T *X, unsigned int nx, unsigned int ny)
{
  for(unsigned int i = 0; i < nx; ++i) {
    for(unsigned int j = 0; j < ny; ++j) {
      int pos = 2 * (i * ny + j);
      std::cout << "(" << X[pos]
		<< "," << X[pos + 1] 
		<< ") ";
    }
    std::cout << std::endl;
  }
}

template<class T>
void show3C(const T *X, unsigned int nx, unsigned int ny, unsigned int nz)
{
  for(unsigned int i = 0; i < nx; ++i) {
    for(unsigned int j = 0; j < ny; ++j) {
      for(unsigned int k = 0; k < nz; ++k) {
	int pos = i * ny * nz + j * nz + k;
	std::cout << "(" << X[2 * pos]
		  << "," << X[2 * pos + 1] 
		  << ") ";
      }
      std::cout << std::endl;
    } 
    std::cout << std::endl;
  }
}

template<class T>
void show1R(const T *X, const unsigned int n)
{
  for(unsigned int i = 0; i < n; ++i)
    std::cout << X[i] << std::endl;
}

template<class T>
void show1R(const T *X, const unsigned int nx, const unsigned int M, 
	    const unsigned int dist = 0)
{
  int dist0 = dist == 0 ? nx : dist;
  for(unsigned int m = 0; m < M; ++m) {
    for(unsigned int i = 0; i < nx; ++i) {
      int pos = m * dist0 + i;
      std::cout << X[pos] << " ";
    }
    std::cout << std::endl;
  }
}

template<class T>
void show2R(const T *X, unsigned int nx, unsigned int ny,
	    unsigned int stride = 0)
{
  if(stride == 0)
    stride = ny;
  for(unsigned int i = 0; i < nx; ++i) {
    for(unsigned int j = 0; j < ny; ++j) {
      unsigned pos = i * stride + j;
      std::cout << X[pos] << " ";
    }
    std::cout << std::endl;
  }
}

template<class T>
void show3R(const T *X, const unsigned int nx, const unsigned int ny,
	    const unsigned int nz, unsigned int skip=0)
{
  if(skip == 0)
    skip = ny;
  for(unsigned int i = 0; i < nx; ++i) {
    for(unsigned int j = 0; j < ny; ++j) {
      for(unsigned int k = 0; k < nz; ++k) {
	unsigned int pos = i * ny * skip + j * skip + k;
	std::cout << X[pos] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

#endif
