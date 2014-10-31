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
vpath %.h ../

all: fft1

clutils.o: ../clutils.c clutils.h
	g++ $(CXXFLAGS) ../clutils.c -c

platform.o: platform.hpp ../platform.cpp
	g++ $(CXXFLAGS) ../platform.cpp -c

fft1.o: fft1.cpp
	g++ $(CXXFLAGS) $^ -c 

fft1: fft1.o platform.o clutils.o
	g++ $^ $(LDFLAGS) -o $@

clean:
	rm -f fft1 *.o
