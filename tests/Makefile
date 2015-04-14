CXX=g++
CC=gcc

INCL=

ifneq ($(strip $(OPENCL_INCLUDE_PATH)),)
INCL+=-I$(OPENCL_INCLUDE_PATH)
endif

ifneq ($(strip $(CLFFT_INCLUDE_PATH)),)
INCL+=-I$(CLFFT_INCLUDE_PATH)
endif

ifneq ($(strip $(FFTWPP_INCLUDE_PATH)),)
INCL+=-I$(FFTWPP_INCLUDE_PATH)
INCL+=-I$(FFTWPP_INCLUDE_PATH)/tests
endif

#ifneq ($(strip $(FFTW_INCLUDE_PATH)),)
INCL+=-I$(FFTW_INCLUDE_PATH)
vpath %.cc $(FFTW_INCLUDE_PATH)
#endif

INCL+=-I.
INCL+=-I..
INCL+=-I$(FFTWPP_INCLUDE_PATH)

vpath %.cc $(FFTWPP_INCLUDE_PATH)
vpath %.cpp ../
vpath %.c ../

# Set up CXXFLAGS
CXXFLAGS=
CXXFLAGS+=-O3
CXXFLAGS+=-Wall
CXXFLAGS+=-fopenmp
CXXFLAGS+=-march=native -mtune=native

CXXFLAGS+=-g -Wall -ansi -fomit-frame-pointer -fstrict-aliasing -ffast-math -mavx -mfpmath=sse
CXXFLAGS+=$(INCL)


# Set up CFLAGS
CFLAGS=$(CXXFLAGS)

MAKEDEPEND=$(CXXFLAGS) -O0 -M -DDEPEND

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

ifneq ($(strip $(FFTW_LIB_PATH)),)
LDFLAGS+=-L$(FFTW_LIB_PATH)
endif

LDLIBS=
LDLIBS+=-lfftw3_omp -lfftw3
LDLIBS+=-lm

# The programs to be produced
OUTPUT=fft1 fft2 fft1r fft2r

SRCS_CPP=clfft platform
SRCS_C=clutils
#SRCS_CC=fftw++
OBJS=$(SRCS_CPP:=.o) $(SRCS_C:=.o) $(SRCS_CC:=.o)
ALL=$(SRCS_CPP) $(SRCS_C) $(SRCS_CC) $(OUTPUT)

all: $(OUTPUT)

fft1: $(OBJS) fft1.o fftw++.o
	@echo compiling $@
	@echo $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

fft1r: $(OBJS) fft1r.o fftw++.o
	@echo compiling $@
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

fft2: $(OBJS) fft2.o fftw++.o
	@echo compiling $@
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

fft2r: $(OBJS) fft2r.o fftw++.o
	@echo compiling $@
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

.cc.o: %.cc %.h %.d
	@echo  $@
	$(CXX) $(CXXFLAGS) $< -c -o $@ $(LDFLAGS) $(LDLIBS)

.cpp.o: %.cpp %.hpp %.d
	@echo  $@
	$(CXX) $(CXXFLAGS) $< -c $(LDFLAGS) $(LDLIBS)

.c.o: %.c %.h %.d
	@echo  $@
	$(CC) $(CFLAGS) $< -c $(LDFLAGS) $(LDLIBS)

.cc.d:
	@echo Creating $@; \
	rm -f $@; \
	${CXX} $(MAKEDEPEND) $(INCL) $< > $@.$$$$ 2>/dev/null && \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

.cpp.d:
	@echo Creating $@; \
	rm -f $@; \
	${CXX} $(MAKEDEPEND) $(INCL) $< > $@.$$$$ 2>/dev/null && \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

.c.d:
	@echo Creating $@; \
	rm -f $@; \
	${CC} $(MAKEDEPEND) $(INCL) $< > $@.$$$$ 2>/dev/null && \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

.SUFFIXES: .c .cc .cpp .o .d

.PHONY: clean all

clean: FORCE
	rm -f *.o *.gch $(OUTPUT) *.d

ifeq (,$(findstring clean,${MAKECMDGOALS}))
-include $(ALL:=.d)
endif

FORCE:
