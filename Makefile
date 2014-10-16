all: clfft1d

CXXFLAGS=

ifneq ($(strip $(CLFFT_INCLUDE_PATH)),)
# erratic:
# export CLFFT_INCLUDE_PATH=${HOME}/clFFT/include
CXXFLAGS+=-I$(CLFFT_INCLUDE_PATH)
endif

ifneq ($(strip $(FFTWPP_INCLUDE_PATH)),)
CXXFLAGS+=-I$(FFTWPP_INCLUDE_PATH)
CXXFLAGS+=-I$(FFTWPP_INCLUDE_PATH)/tests
endif

LDFLAGS=
LDFLAGS+=-lOpenCL
ifneq ($(strip $(CLFFT_LIB_PATH)),)
# erratic:
# export CLFFT_LIB_PATH=${HOME}/clFFT/lib
LDFLAGS+=-L$(CLFFT_LIB_PATH)
endif
LDFLAGS+=-lclFFT

clfft.o: clfft.cpp
	g++ $(CXXFLAGS) clfft.cpp -c

clfft1d: clfft.o 
	g++ clfft.o $(LDFLAGS) -o clfft1d

clean:
	rm -f clfft.o clfft1d
