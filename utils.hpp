#ifndef __utils_hpp__
#define __utils_hpp__ 1

#include <iostream>

void usage(unsigned int dim) 
{
  std::cout << "usage:\n"
	    << "./clfft1 \n"
	    << "\t-p <int>\tOpenCL platform number \n"
	    << "\t-d <int>\tOpenCL device number \n"
	    << "\t-c <0 or 1>\tinclude memory copy in time \n"
	    << "\t-m <int>\tproblem size \n"
	    << "\t-x <int>\tproblem size in first dimension \n";
  if(dim > 1)
    std::cout <<  "\t-y <int>\tproblem size in second dimension\n";
  std::cout << "\t-N <int>\tNumber of tests \n"
	    << "\t-S <int>\tStatistical measure to use \n"
	    << std::endl;
}


template<class T>
void show1C(const T *X, int n)
{
  for(int i = 0; i < n; ++i) {
    std::cout << "(" 
	      << X[2 * i] 
	      << "," 
	      <<  X[2 * i +1] 
	      << ")" 
	      << std::endl;
  }
}

template<class T>
void show1R(const T *X, int n)
{
  for(unsigned int i = 0; i < n; ++i)
    std::cout << X[i] << std::endl;
}


template<class T>
void show2C(const T *X, int nx, int ny)
{
  for(unsigned int i=0; i < nx; ++i) {
    for(unsigned int j=0; j < ny; ++j) {
      unsigned pos = i*ny + j; 
      std::cout << "(" << X[2*pos] 
		<< "," << X[2*pos +1] 
		<< ") ";
    }
    std::cout << std::endl;
  }
}

#endif
