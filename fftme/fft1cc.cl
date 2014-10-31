__kernel void fft1cc(int n, __global float *f)
{
  /* const unsigned int l2n=log2(n); */
  
  int i;
  for(i=0; i < n; ++i) {
    f[2*i]=2*i;
    f[2*i+1]=2*i+1;
  }
  
  f[0]=3.4;
}
