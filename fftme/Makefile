CXXFLAGS=

CXXFLAGS+=-std=c++0x

ifneq ($(strip $(OPENCL_INCLUDE_PATH)),)
CXXFLAGS+=-I$(OPENCL_INCLUDE_PATH)
endif

CXXFLAGS+=-I.

CXXFLAGS+=-I..

CXXFLAGS+=-Ofast
CXXFLAGS+=-Wall 

LDFLAGS=

LDFLAGS+=-lOpenCL
ifneq ($(strip $(OPENCL_LIB_PATH)),)
LDFLAGS+=-L$(OPENCL_LIB_PATH)
endif

vpath %.hpp ../

all: fft1

platform.o: platform.hpp ../platform.cpp
	g++ $(CXXFLAGS) ../platform.cpp -c

fft1.o: fft1.cpp
	g++ $(CXXFLAGS) $^ -c 

fft1: fft1.o platform.o 
	g++ $^ $(LDFLAGS) -o $@

clean:
	rm -f fft1 *.o
