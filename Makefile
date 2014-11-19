

CXXFLAGS=

ifneq ($(strip $(OPENCL_INCLUDE_PATH)),)
CXXFLAGS+=-I$(OPENCL_INCLUDE_PATH)
endif

ifneq ($(strip $(CLFFT_INCLUDE_PATH)),)
# erratic:
# export CLFFT_INCLUDE_PATH=${HOME}/clFFT/include
CXXFLAGS+=-I$(CLFFT_INCLUDE_PATH)
endif

ifneq ($(strip $(FFTWPP_INCLUDE_PATH)),)
CXXFLAGS+=-I$(FFTWPP_INCLUDE_PATH)
CXXFLAGS+=-I$(FFTWPP_INCLUDE_PATH)/tests
endif



CXXFLAGS+=-I.

CXXFLAGS+=-O3
CXXFLAGS+=-Wall

LDFLAGS=

LDFLAGS+=-lOpenCL
ifneq ($(strip $(OPENCL_LIB_PATH)),)
LDFLAGS+=-L$(OPENCL_LIB_PATH)
endif

LDFLAGS+=-lclFFT
ifneq ($(strip $(CLFFT_LIB_PATH)),)
LDFLAGS+=-L$(CLFFT_LIB_PATH)
endif

LDFLAGS+=-lpthread 



all: fft1 fft2 fft1r

clutils.o: clutils.c clutils.h
	$(CXX) $(CXXFLAGS) clutils.c -c

platform.o: platform.hpp platform.cpp
	$(CXX) $(CXXFLAGS) platform.cpp -c

clfft.o: clfft.cpp clfft.hpp
	$(CXX) $(CXXFLAGS) clfft.cpp  -c 

fft1.o: fft1.cpp utils.hpp
	$(CXX) $(CXXFLAGS) $^ -c 

t2.o: fft2.cpp utils.hpp
	$(CXX) $(CXXFLAGS) $^ -c 

fft1r.o: fft1r.cpp utils.hpp
	$(CXX) $(CXXFLAGS) $^ -c 

fft1: fft1.o clfft.o platform.o clutils.o 
	$(CXX) $^ $(LDFLAGS) -o $@

fft2: fft2.o clfft.o platform.o clutils.o
	$(CXX) $^ $(LDFLAGS) -o $@

fft1r: fft1r.o clfft.o platform.o clutils.o
	$(CXX) $^ $(LDFLAGS) -o $@


clean:
	rm -f *.o *.gch fft1 fft2 fft1r
