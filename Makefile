

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

CXXFLAGS+=-Ofast
CXXFLAGS+=-Wall

LDFLAGS=

LDFLAGS+=-lOpenCL
ifneq ($(strip $(OPENCL_LIB_PATH)),)
LDFLAGS+=-L$(OPENCL_LIB_PATH)
endif

ifneq ($(strip $(CLFFT_LIB_PATH)),)
LDFLAGS+=-L$(CLFFT_LIB_PATH)
endif
LDFLAGS+=-lclFFT

all: clfft1 clfft2 clfft1r

clutils.o: clutils.c clutils.h
	g++ $(CXXFLAGS) clutils.c -c

platform.o: platform.hpp platform.cpp
	g++ $(CXXFLAGS) platform.cpp -c

clfft.o: clfft.cpp clfft.hpp
	g++ $(CXXFLAGS) clfft.cpp  -c 

clfft1.o: clfft1.cpp utils.hpp
	g++ $(CXXFLAGS) $^ -c 

clfft2.o: clfft2.cpp utils.hpp
	g++ $(CXXFLAGS) $^ -c 

clfft1r.o: clfft1r.cpp utils.hpp
	g++ $(CXXFLAGS) $^ -c 

clfft1: clfft1.o clfft.o platform.o clutils.o 
	g++ $^ $(LDFLAGS) -o $@

clfft2: clfft2.o clfft.o platform.o clutils.o
	g++ $^ $(LDFLAGS) -o $@

clfft1r: clfft1r.o clfft.o platform.o clutils.o
	g++ $^ $(LDFLAGS) -o $@


clean:
	rm -f *.o clfft1 clfft2 clfft1r
