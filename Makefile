CXX=g++
CC=gcc

# Set up CXXFLAGS
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

# Set up CCFLAGS
CCFLAGS=$(CXXFLAGS)

# set up LDFLAGS
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

# Define the source files and objects
SRCS_CPP=clfft.cpp platform.cpp utils.cpp
SRCS_CC= 
SRCS_C=clutils.c
CPPOBJS=$(SRCS_CPP:.cpp=.o)
CCOBJS+= $(SRCS_CC:.cc=.o)
COBJS=$(SRCS_C:.c=.o)
OBJS=$(CPPOBJS) $(CCOBJS) $(COBJS) 

# The programs to be produced
output=fft1 fft2 fft1r

all: $(output)

# static pattern rule
$(output) : % : %.o $(OBJS)
	@echo $^
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o : %.cpp %.hpp
	@echo $@
	$(CXX) -c $(CXXFLAGS) $<
	$(CXX) -MM $(CXXFLAGS) $*.cpp > $*.d

%.o : %.cc %.h
	@echo $@
	$(CXX) -c $(CXXFLAGS) $<
	$(CXX) -MM $(CXXFLAGS) $*.cc > $*.d

%.o : %.c %.h
	@echo $@
	$(CC) -c $(CCFLAGS) $<
	$(CC) -MM $(CFLAGS) $*.c > $*.d

.SUFFIXES+= .d

clean:
	rm -f *.o *.gch $(output) *.d
