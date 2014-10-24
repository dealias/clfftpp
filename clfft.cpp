#include <clfft.hpp>

void clfft_setup()
{
  cl_int err;  
  clfftSetupData fftSetup;
  err = clfftInitSetupData(&fftSetup);
  err = clfftSetup(&fftSetup);
}
