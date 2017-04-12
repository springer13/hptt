CXX_FLAGS=-O3 -std=c++11 -fPIC

ifeq ($(CXX),icpc)
CXX_FLAGS += -qopenmp -xhost 
else
ifeq ($(CXX),g++)
CXX_FLAGS += -fopenmp -march=native 
else
ifeq ($(CXX),clang++)
CXX_FLAGS += -fopenmp
endif
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

SRC=$(wildcard ./src/*.cpp)
OBJ=$(SRC:.cpp=.o)

all: ${OBJ}
	${CXX} ${OBJ} ${CXX_FLAGS} -o lib/libhptt.so -shared

%.o: %.cpp
	${CXX} ${CXX_FLAGS} ${INCLUDE_PATH} -c $< -o $@

clean:
	rm -rf src/*.o lib/libhptt.so
