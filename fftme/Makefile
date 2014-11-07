CXXFLAGS=

CXXFLAGS+=-std=c++0x

ifneq ($(strip $(OPENCL_INCLUDE_PATH)),)
CXXFLAGS+=-I$(OPENCL_INCLUDE_PATH)
endif

ifneq ($(strip $(FFTWPP_INCLUDE_PATH)),)
CXXFLAGS+=-I$(FFTWPP_INCLUDE_PATH)
CXXFLAGS+=-I$(FFTWPP_INCLUDE_PATH)/tests
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

all: mfft1

clutils.o: ../clutils.c clutils.h
	g++ $(CXXFLAGS) ../clutils.c -c

platform.o: platform.hpp ../platform.cpp
	g++ $(CXXFLAGS) ../platform.cpp -c

mfft1.o: mfft1.cpp
	g++ $(CXXFLAGS) $^ -c 

mfft1: mfft1.o platform.o clutils.o
	g++ $^ $(LDFLAGS) -o $@

clean:
	rm -f mfft1 *.o
