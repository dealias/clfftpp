

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

LDFLAGS=

LDFLAGS+=-lOpenCL
ifneq ($(strip $(OPENCL_LIB_PATH)),)
LDFLAGS+=-L$(OPENCL_LIB_PATH)
endif

ifneq ($(strip $(CLFFT_LIB_PATH)),)
LDFLAGS+=-L$(CLFFT_LIB_PATH)
endif
LDFLAGS+=-lclFFT

all: clfft1

platform.o: platform.hpp platform.cpp
	g++ $(CXXFLAGS) platform.cpp -c

clfft.o: clfft.cpp clfft.hpp
	g++ $(CXXFLAGS) clfft.cpp  -c 

clfft1.o: clfft1.cpp clfft1.hpp
	g++ $(CXXFLAGS) clfft1.cpp  -c 

clfft1: clfft1.o clfft.o platform.o 
	g++ $^ $(LDFLAGS) -o $@

clean:
	rm -f *.o clfft1
