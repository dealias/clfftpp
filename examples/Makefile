CXX=g++
CC=gcc

INCL=

ifneq ($(strip $(OPENCL_INCLUDE_PATH)),)
INCL+=-I$(OPENCL_INCLUDE_PATH)
endif

ifneq ($(strip $(CLFFT_INCLUDE_PATH)),)
INCL+=-I$(CLFFT_INCLUDE_PATH)
endif

INCL+=-I.
INCL+=-I..

vpath %.cpp ../
vpath %.c ../

# Set up CXXFLAGS
CXXFLAGS=
CXXFLAGS+=-O3
CXXFLAGS+=-Wall

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

# The programs to be produced
OUTPUT=fft1 fft2 fft1r fft2r

SRCS_CPP=clfft platform
SRCS_C=clutils
OBJS=$(SRCS_CPP:=.o) $(SRCS_C:=.o)
ALL=$(SRCS_CPP) $(SRCS_C) $(OUTPUT)

all: $(OUTPUT)

fft1: $(OBJS) fft1.o
	$(CXX) -o $@ $^ $(LDFLAGS)

fft1r: $(OBJS) fft1r.o
	$(CXX) -o $@ $^ $(LDFLAGS)

fft2: $(OBJS) fft2.o
	$(CXX) -o $@ $^ $(LDFLAGS)

fft2r: $(OBJS) fft2r.o
	$(CXX) -o $@ $^ $(LDFLAGS)

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

.SUFFIXES: .c .cpp .o .d

.PHONY: clean
clean: FORCE
	rm -f *.o *.gch $(OUTPUT) *.d

ifeq (,$(findstring clean,${MAKECMDGOALS}))
-include $(ALL:=.d)
endif

FORCE: