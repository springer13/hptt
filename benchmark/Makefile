CXX_FLAGS=-O3 -std=c++14 -fPIC

ifeq ($(CXX),icpc)
CXX_FLAGS += -qopenmp -xhost 
else
ifeq ($(CXX),g++)
CXX_FLAGS += -fopenmp -march=native 
endif
endif

avx: 
	${MAKE} clean 
	${MAKE} avx2
arm: 
	${MAKE} clean 
	${MAKE} arm2
scalar: 
	${MAKE} clean 
	${MAKE} scalar2

avx2:CXX_FLAGS+=-DHPTT_ARCH_AVX
avx2: all
arm2: CXX_FLAGS+=-mfpu=neon -DHPTT_ARCH_ARM
arm2: all
scalar2: all

SRC=benchmark.cpp reference.cpp
OBJ=$(SRC:.cpp=.o)

INCLUDE_PATH=-I../src/ 
LIB_PATH=-L../lib/
LIBS=-lhptt

all: ${OBJ}
	${CXX} ${OBJ} ${LIB_PATH} ${LIBS} ${CXX_FLAGS} -o benchmark.exe

%.o: %.cpp
	${CXX} ${CXX_FLAGS} ${INCLUDE_PATH} -c $< -o $@

clean:
	rm -rf *.o benchmark.exe
