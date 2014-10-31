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
  const int kmax=n/2;
  unsigned int twoj=n/2;
  for(unsigned int j=0; j < l2n; ++j) {
    for(unsigned int k=0; k < kmax; ++k) {

      unsigned int ke, ko;
      uint2binary(k,kb,l2n-1);      
      ke = even(l2n, j, kb);
      ko = ke + twoj;
      uint2binary(ke,kb,l2n);
      uint2binary(ko,kb,l2n);

      double xe[2]={x[2*ke],x[2*ke+1]};
      double xo[2]={x[2*ko],x[2*ko+1]};
      
      double w[2]={cos(-2.0*PI*k*j/(double)n),sin(-2.0*PI*k*j/(double)n)};
      
      double temp[2]={xo[0]*w[0] - xo[1]*w[1],
		      xo[0]*w[1] + xo[2]*w[1]};
      
      x[2*ke]   = xe[0] + temp[0];
      x[2*ke+1] = xe[1] + temp[1];
      x[2*ko]   = xe[0] - temp[0];
      x[2*ko+1] = xe[1] - temp[1];
    }
    twoj /= 2;
  }

  delete kb;
}

int main()
{
  unsigned int n=1024; 
 
  double *x=new double[2*n];

  for(unsigned int i=0; i < n; ++i) {
    x[2*i] =i;
    x[2*i+1] =0.0;
  }
  
  if(n < 100) {
    std::cout << "input:" << std::endl;
    for(unsigned int i=0; i < n; ++i) {
      std::cout << "(" << x[2*i] 
		<< "," << x[2*i+1]
		<< ")" << std::endl;
    }
  }

  fft(x,n);

  if(n < 100) {
    std::cout << "output:" << std::endl;
    for(unsigned int i=0; i < n; ++i) {
      std::cout << "(" << x[2*i] 
		<< "," << x[2*i+1]
		<< ")" << std::endl;
    }
  }

  delete x;
  
}


