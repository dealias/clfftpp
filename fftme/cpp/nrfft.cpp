#include <iostream>
#include <math.h>

const double PI=4.0*atan(1.0);

unsigned int log2(unsigned int n) 
{
  unsigned int r = 0; // r will be lg(v)
  while (n >>= 1)
    r++;
  return r;
}

void uint2binary(unsigned int n, unsigned int *bn, const unsigned int l2n) 
{
  for (unsigned int i = 0; i < l2n; ++i) {
    bn[i] = n & 1;
    n /= 2;
  }
}


unsigned int revbin(unsigned int x, const unsigned int n)
{
  unsigned int j=0;
  unsigned int ldn = log2(n); // is an integer
  while (ldn > 0) {
    j = j << 1;
    j = j + (x & 1);
    x = x >> 1;
    --ldn;
  }
  return j;
}

inline unsigned revbin_update(unsigned r, unsigned n)
{
  // for (unsigned m=n>>1; (!((r^=m)&m)); m>>=1);
  // return r;

  do {
    std::cout << "n: " << n  << std::endl;
    std::cout << "r: " << r  << std::endl;
    n = n >> 1;
    r = r^n;
  } while ((r&n) == 0);
  return r;
}

void swap(double *a, const unsigned int x, const unsigned int r)
{
  double t[2]={a[2*x],a[2*x+1]};
  a[2*x]=a[2*r];
  a[2*x+1]=a[2*r+1];
  a[2*r]=t[0];
  a[2*r+1]=t[1];
}

void revbin_permute(double *a, const unsigned int n)
// a[0..n-1] input,result
{
  if(n <= 2) 
    return;
  unsigned int r=0;
  for(unsigned int x=0; x < n; ++x) {
    unsigned int n0=n;
    unsigned int r = revbin_update(r, n0); // inline me
    if(r > x) swap(a,x,r);
  }
}


void bit_reverse_reorder(double *W, int N)
{
  int bits, i, j, k;
  double tempr, tempi;

#define MAXPOW 24
    
int pow_2[MAXPOW];
 pow_2[0]=1;
 for (i=1; i<MAXPOW; i++)
   pow_2[i]=pow_2[i-1]*2;
 
 for (i=0; i<MAXPOW; i++)
    if (pow_2[i]==N) bits=i;

  for (i=0; i<N; i++)
    {
      j=0;
      for (k=0; k<bits; k++)
	if (i&pow_2[k]) j+=pow_2[bits-k-1];
      
      if (j>i)  /** Only make "up" swaps */
	{
	  tempr=W[2*i];
	  tempi=W[2*i+1];
	  W[2*i]=W[2*j];
	  W[2*i+1]=W[2*j+1];
	  W[2*j]=tempr;
	  W[2*j+1]=tempi;
	}
    }
}


// void revbin_permute(double *a, unsigned int n)
// {
//   for (unsigned int x=0; x < n; ++x) {
//     unsigned int r = revbin(x, n);
//     if(r > x) {
      
//       //then swap(a[x], a[r])
//     }
//   }
// }


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
    //std::cout << "\t\t\ti: " << i << "\tpkb: " << pkb << std::endl;
    ke += pkb;
    p *= 2;
  }

  p *= 2;

  for (unsigned int i = j0; i < l2nm1; ++i) {
    unsigned int pkb=p*kb[i];
    //std::cout << "\t\t\ti: " << i << "\tpkb: " << pkb << std::endl;
    ke += pkb;
    p *= 2;
  }

  return ke;
}


void fft(double *x, unsigned int n) {
  const unsigned int l2n=log2(n);
  unsigned int *kb=new unsigned int[l2n];
  const unsigned int kmax=n/2;
  unsigned int twoj=n/2;
  for(unsigned int j=0; j < l2n; ++j) {
    std::cout << "j: " << j << std::endl; 
    for(unsigned int k=0; k < kmax; ++k) {

      std::cout << "\tk: " << k << std::endl; 
    
      unsigned int ke, ko;
      uint2binary(k,kb,l2n-1);
      ke = even(l2n, j, kb);
      ko = ke + twoj;
      std::cout << "\t\tke: " << ke << std::endl; 
      std::cout << "\t\tko: " << ko << std::endl; 
      std::cout << "\t\ttwoj: " << twoj << std::endl; 

      double xe[2]={x[2*ke],x[2*ke+1]};
      double xo[2]={x[2*ko],x[2*ko+1]};

      //double arg=-2.0*PI*k*j/(double)n;
      double arg=-2.0*PI*ke/(double)(2.0*twoj);

      double w[2]={cos(arg),sin(arg)};
      std::cout << "\t\tw: (" << w[0] << "," << w[1] << ")" << std::endl;

      x[2*ke]   = xe[0] + xo[0];
      x[2*ke+1] = xe[1] + xo[1];

      double t[2]={xe[0] - xo[0], xe[1] - xo[1]};
      
      x[2*ko]   = w[0]*t[0] - w[1]*t[1];
      x[2*ko+1] = w[1]*t[0] + w[0]*t[1];
    }
    twoj /= 2;
  }

  delete kb;

  // FIXME: unshuffle x.
}

void ffta(double *x, unsigned int n) {
  const unsigned int ldn=log2(n);
  unsigned int is=1; // sign

  for(unsigned int ldm=ldn; ldm > 0; --ldm) {
    std::cout << ldm << std::endl;
    unsigned int m=1;
    for(unsigned int i=0; i < ldm; ++i)
      m *= 2;
    unsigned int mh = m/2;
    std::cout << "mh: " << mh << std::endl;
    for(unsigned int j=0; j < mh; ++j) {
      std::cout << "\tj: " << j << std::endl;
      double w[2]={cos(-2.0*PI*j/(double)m),sin(-2.0*PI*j/(double)m)};
      //e := exp(is*2*PI*I*j/m)
      for(unsigned int r=0; r < n-1; r += m) {
	std::cout << "\t\tr: " << r << std::endl;
	double u[2] = {x[2*(r+j)],x[2*(r+j)+1]};
	double v[2] = {x[2*(r+j+mh)],x[2*(r+j+mh)+1]};
	x[2*(r+j)] = u[0] + v[0];
	x[2*(r+j)+1] = u[1] + v[1];
	double t[2]={u[0] - v[0], u[1] + v[1]};
	x[2*(r+j+mh)]=t[0]*w[0] - t[1]*w[1];
	x[2*(r+j+mh)+1]=t[0]*w[1] + t[1]*w[0];
      }
    }
    //revbin_permute(x,n);

    bit_reverse_reorder(x, n);
  }
}

int main()
{
  unsigned int n=4;//1024; 
 
  double *x=new double[2*n];

  for(unsigned int i=0; i < n; ++i) {
    x[2*i] =0;
    x[2*i+1] =i ; //0.0;
  }
  
  if(n < 100) {
    std::cout << "input:" << std::endl;
    for(unsigned int i=0; i < n; ++i) {
      std::cout << "(" << x[2*i] 
		<< "," << x[2*i+1]
		<< ") ";
    }
    std::cout << std::endl;
  }

  fft(x,n);

  if(n < 100) {
    std::cout << "output:" << std::endl;
    for(unsigned int i=0; i < n; ++i) {
      std::cout << "(" << x[2*i] 
		<< "," << x[2*i+1]
		<< ") ";
    }
    std::cout << std::endl;
  }

  delete x;
  
}
