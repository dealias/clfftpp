__kernel void fft1cc(int nx, int ny, __global float *f)
{
  /* const unsigned int l2n=log2(n); */

  const int idx = get_global_id(0);

  const int ix=idx;
  int iy;
  for(iy=0; iy < ny; ++iy) {
    int pos=2*(idx*ny + iy);
    f[pos]=idx;
    f[pos+1]=iy+1;
  }

}

