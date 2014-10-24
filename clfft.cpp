/* No need to explicitely include the OpenCL headers */
#include <clFFT.h>

void clfft_setup()
{
  cl_int err;  
  clfftSetupData fftSetup;
  err = clfftInitSetupData(&fftSetup);
  err = clfftSetup(&fftSetup);
}
