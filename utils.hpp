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
    std::cout <<  "\t-u <int>\tproblem size in second dimension\n";
  std::cout << "\t-N <int>\tNumber of tests \n"
	    << "\t-S <int>\tStatistical measure to use \n"
	    << std::endl;
}

