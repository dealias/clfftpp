CXXFLAGS=

CXXFLAGS+=-std=c++0x

ifneq ($(strip $(OPENCL_INCLUDE_PATH)),)
CXXFLAGS+=-I$(OPENCL_INCLUDE_PATH)
endif

ifneq ($(strip $(FFTWPP_INCLUDE_PATH)),)
CXXFLAGS+=-I$(FFTWPP_INCLUDE_PATH)
CXXFLAGS+=-I$(FFTWPP_INCLUDE_PATH)/tests
endif

ifneq ($(strip $(FFTW_INCLUDE_PATH)),)
CXXFLAGS+=-I$(FFTW_INCLUDE_PATH)
endif

ifneq ($(strip $(FFTW_LIB_PATH)),)
LDFLAGS+=-L$(FFTW_LIB_PATH)
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

LDFLAGS += -lfftw3
LDFLAGS += -lm

vpath %.hpp ../
vpath %.h ../
vpath %.h $(FFTWPP_INCLUDE_PATH)


all: mfft1

fftw++.o: $(FFTWPP_INCLUDE_PATH)/fftw++.cc $(FFTWPP_INCLUDE_PATH)/fftw++.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

fft.o: fft.hpp fft.cpp 
	g++ $(CXXFLAGS) fft.cpp -c

clutils.o: ../clutils.c clutils.h
	g++ $(CXXFLAGS) ../clutils.c -c

platform.o: platform.hpp ../platform.cpp
	g++ $(CXXFLAGS) ../platform.cpp -c

mfft1.o: mfft1.cpp
	g++ $(CXXFLAGS) $^ -c

mfft1: mfft1.o platform.o clutils.o fft.o fftw++.o
	g++ mfft1.o platform.o clutils.o fftw++.o $(LDFLAGS) -o $@

clean:
	rm -f mfft1 *.o
