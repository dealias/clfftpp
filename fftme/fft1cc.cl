unsigned int uintlog2(unsigned int n)
{
  unsigned int r = 0;
  while (n >>= 1)
    r++;
  return r;
}

void swap(__global float *f, const unsigned int a, const unsigned int b)
{
  float temp[2]={f[2*a],f[2*a+1]};
  f[2*a]   = f[2*b];
  f[2*a+1] = f[2*b+1];
  f[2*b]   = temp[0];
  f[2*b+1] = temp[1];
}

unsigned int binary2uint(unsigned int kb[], const unsigned int l2n)
{
  unsigned int k=0;

  unsigned int p=1;
  for(unsigned int i=0; i < l2n; ++i) {
    k += kb[i] * p;
    p *= 2;
  }
  return k;
}

void uint2binary(unsigned int n, unsigned int *bn, const unsigned int l2n) 
{
  unsigned int i;
  for (i = 0; i < l2n; ++i) {
    bn[i] = n & 1;
    n /= 2;
  }
}

unsigned int bitreverse(const unsigned int k, const unsigned int log2ny)
{
  unsigned int kb[32]; // this is too big, but it compiles!
  uint2binary(k,kb,log2ny);
  for(unsigned int i=1; i < log2ny; ++i)
    kb[i] = ! kb[i];
  return binary2uint(kb,log2ny);
}

/* unsigned int bitreverse(unsigned int i, unsigned int size) */
/* { */
/*   long result,mask; */

/*   result=0; */
/*   for(; size > 1; size>>= 1){ */
/*     mask=i&1; */
/*     i>>=1; */
/*     result<<=1; */
/*     result|=mask; */
/*   } */
/* } */

void unshuffle( __global float *fx, const unsigned int ny)
{
  const unsigned int log2ny=uintlog2(ny);
  
  // FIXME: this is not working.

  for(unsigned int k=0; k < ny; ++k) {
    unsigned int j=bitreverse(k,log2ny);
    //unsigned int j=bitreverse(k,ny);
    if(j < k) swap(fx,j,k);
    //fx[2*k]=j;
  }
}



unsigned int even(const unsigned int l2n, 
		  const unsigned int j, 
		  const unsigned int *kb)
{
  const unsigned int l2nm1=l2n-1;
  const unsigned int j0=l2nm1-j;

  unsigned int ke=0;
  unsigned int p=1;
 
  for (unsigned int i = 0; i < j0; ++i) {
    unsigned int pkb=p*kb[i];
    ke += pkb;
    p *= 2;
  }

  p *= 2; // Skip one power of two

  for (unsigned int i = j0; i < l2nm1; ++i) {
    unsigned int pkb=p*kb[i];
    ke += pkb;
    p *= 2;
  }
  return ke;
}

__kernel void fft1cc(unsigned int nx, unsigned int ny, __global float *f)
{
  /* const unsigned int l2n=log2(n); */

  const unsigned int idx = get_global_id(0);
  float *fx=f+2*(idx*ny);

  const unsigned int offset=2*(idx*ny);

  const unsigned int log2ny=uintlog2(ny);
  
  const float PI=4.0*atan(1.0);
  //const unsinged int nbits=32;
  unsigned int kb[32]; // this is too big, but it compiles!
  /* unsigned int *kb=new unsigned int[log2ny]; */
  const unsigned int kymax = ny / 2;
  
  unsigned int twojy = ny / 2;
  for(unsigned int iy = 0; iy < log2ny; ++iy) {
    for(unsigned int ky = 0; ky < kymax; ++ky) {
      uint2binary(ky,kb,log2ny-1);
	
      const unsigned int ke = even(log2ny, iy, kb);
      const unsigned int ko = ke + twojy;

      float fe[2] = {f[offset+2*ke], f[offset+2*ke+1]};
      float fo[2] = {f[offset+2*ko], f[offset+2*ko+1]};
      
      /* const float arg = -2.0 * PI * ky * iy / (float)ny; */
      const float arg = -2.0 * PI * ke / (2.0 * twojy);
      const float w[2] = {cos(arg), sin(arg)};
      
      f[offset+2*ke]   = fe[0] + fo[0];
      f[offset+2*ke+1] = fe[1] + fo[1];

      float t[2]={fe[0] - fo[0], fe[1] - fo[1]};
      
      f[offset+2*ko]   = w[0]*t[0] - w[1]*t[1];
      f[offset+2*ko+1] = w[1]*t[0] + w[0]*t[1];
    }
    twojy /= 2;
  }

  /* FIXME: unshuffle fx */
  unshuffle(fx,ny);
  

  
}

