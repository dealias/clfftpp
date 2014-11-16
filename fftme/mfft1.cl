/* #pragma OPENCL EXTENSION cl_khr_fp64: enable */
/* #define REAL double */
//#define REAL float

#include "fft_double.cl" // defines REAL as float or double.

unsigned int uintlog2(unsigned int n)
{
  unsigned int r = 0;
  while (n >>= 1)
    r++;
  return r;
}

void swap(__local REAL *f, const unsigned int a, const unsigned int b)
{
  const unsigned int twoa = 2 * a;
  const unsigned int twob = 2 * b;
  REAL temp[2]={f[twoa],f[twoa+1]};
  f[twoa]     = f[twob];
  f[twoa + 1] = f[twob + 1];
  f[twob]     = temp[0];
  f[twob + 1] = temp[1];
}

unsigned int bitreverse(const unsigned int k, const unsigned int log2ny)
{
  //http://www.katjaas.nl/bitreversal/bitreversal.html
  /* unsigned int n=k; */
  /* unsigned int bits=log2ny; */

  /* unsigned int nrev, N; */
  /* unsigned int count;    */
  /* N = 1<<bits; */
  /* count = bits-1;   // initialize the count variable */
  /* nrev = n; */
  /* for(n>>=1; n; n>>=1) */
  /*   { */
  /*     nrev <<= 1; */
  /*     nrev |= n & 1; */
  /*     count--; */
  /*   } */
  /* nrev <<= count; */
  /* nrev &= N - 1; */
  /* return nrev; */

  //http://www.openclblog.com/2014/11/fft-concern.html
  unsigned int x = k;
  unsigned int numBits = log2ny;
  unsigned int ans = x & 1;
  while(--numBits) {
    x >>= 1;
    ans <<= 1;
    ans += x & 1;
  }
  return ans;

  // The really slow way:
  /* unsigned int kb[32]; // this is too big, but it compiles! */
  /* uint2binary(k,kb,log2ny); */
  /* unsigned int kr=0; */
  /* unsigned int p=1; */
  /* for(unsigned int i=log2ny; i-- > 0;) { */
  /*   kr += kb[i] * p; */
  /*   p *= 2; */
  /* } */
  /* return kr; */
}

// TODO: can we do the unshuffle and the copy-to-global at the same time?
void unshuffle(__local REAL *lfx, const unsigned int ny)
{
  const unsigned int log2ny = uintlog2(ny);
  for(unsigned int k = 0; k < ny; ++k) {
    unsigned int j = bitreverse(k, log2ny);
    if(j < k) swap(lfx, j, k);
  }
}

unsigned int keven(unsigned int j, unsigned int k)
{
  unsigned int kb=0;
  for(unsigned int jj=0; jj <= j; ++jj)
    kb = (kb << 1) +1;

  return ((k  & ~kb) << 1 ) ^ kb;
}

__kernel
void set_zbuf(unsigned int n, 
	      __global REAL *lz)
{
  const REAL PI = 4.0 * atan(1.0);
  for(unsigned int i = 0; i < n; ++i) {
    const REAL arg = -2.0 * PI * i / (REAL) n;
    lz[2 * i] = cos(arg);
    lz[2 * i + 1] = sin(arg);
  }
}

__kernel 
void mfft1(unsigned int nx,
	   unsigned int mx, 
	   unsigned int ny, 
	   unsigned int stride, 
	   unsigned int dist, 
	   __global REAL *f,
	   __local REAL *lf,
	   __constant REAL *lz // TODO: make __constant
	   )
{
  /* const unsigned int l2n=log2(n); */

  const unsigned int idx = get_global_id(0);

  const unsigned int log2ny = uintlog2(ny);
  
  const unsigned int kymax = ny >> 1;
 
  // Loop from idx to idx+mx
  const unsigned int ixstart = mx * idx;
  const unsigned int ixstop = min(ixstart + mx, nx);
  for(unsigned int ix = ixstart; ix < ixstop; ++ix) {

    __local REAL *lfx = lf + 2 * idx * ny;
    
    /* // Copy to local memory */

    /* Without stride */
    /* __global REAL *fx = f + 2 * ix * ny; */
    /* for(unsigned int iy = 0; iy < 2*ny; ++iy) */
    /*   lfx[iy] = fx[iy]; */

    /* With stride */
    for(unsigned int iy=0; iy < ny; ++iy) {
      unsigned int lpos = 2 * iy;
      unsigned int rpos = 2 * (idx * dist + iy * stride);
      lfx[lpos] = f[rpos];
      lfx[lpos + 1] = f[rpos + 1];
    }

    for(unsigned int iy = 0; iy < log2ny; ++iy) {
      
      unsigned int mask = 0;
      for(unsigned int jj=0; jj < log2ny - 1 - iy; ++jj)
	mask = (mask << 1) +1;

      for(unsigned int ky = 0; ky < kymax; ++ky) {

	const unsigned int ke = (((ky & ~mask) << 1) | (ky & mask)) << 1;
	const unsigned int ko = (ke | (1 << (log2ny - iy)));

	const REAL fe[2] = {lfx[ke], lfx[ke+1]};
	const REAL fo[2] = {lfx[ko], lfx[ko+1]};
      
	//unsigned int kk = (ke << (iy+1))>>2;
	//const REAL arg = -2.0 * PI * kk / (REAL) ny;
	//const REAL w[2] = {cos(arg), sin(arg)};

	const unsigned int kk = (((ke << (iy+1))>>2)%ny) << 1;
	const REAL w[2] = {lz[kk], lz[kk+1]};
	//printf("%d\n",kk);
	//printf("w[0]: %f, zl[%d]: %f,\t%f\n",w[0],kk/2,lz[2*kk],w[0]-lz[kk]);
		
	lfx[ke]   = fe[0] + fo[0];
	lfx[ke+1] = fe[1] + fo[1];

	REAL t[2] = {fe[0] - fo[0], fe[1] - fo[1]};
      
	lfx[ko]   = w[0]*t[0] - w[1]*t[1];
	lfx[ko+1] = w[1]*t[0] + w[0]*t[1];
      }

    }


    // Bit-reversal stage
    unshuffle(lfx, ny);

    // Copy from local memory to global memory

    /* Without stride: */
    /* for(unsigned int iy = 0; iy < 2 * ny; ++iy) { */
    /*   fx[iy] = lfx[iy]; */
    /* } */

    /* With stride: */
    for(unsigned int iy=0; iy < ny; ++iy) {
      unsigned int lpos = 2 * iy;
      unsigned int rpos = 2*(idx * dist + iy * stride);
      f[rpos] = lfx[lpos];
      f[rpos + 1] = lfx[lpos + 1];
    }

  }
}
