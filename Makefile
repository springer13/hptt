CXX_FLAGS=-O3 -std=c++11 -DNDEBUG
INCLUDE_PATH=-I./include/

ifeq ($(CXX),icpc)
CXX_FLAGS += -qopenmp -xhost 
else
ifeq ($(CXX),g++)
CXX_FLAGS += -fopenmp -mcpu=native 
else
ifeq ($(CXX),clang++)
CXX_FLAGS += -fopenmp -march=native
endif
endif
endif

avx: 
	${MAKE} clean 
	${MAKE} avx2
arm: 
	${MAKE} clean 
	${MAKE} arm2
ibm: 
	${MAKE} clean 
	${MAKE} ibm2
scalar: 
	${MAKE} clean 
	${MAKE} scalar2

avx2:CXX_FLAGS+=-mavx -DHPTT_ARCH_AVX
avx2: all
arm2: CXX_FLAGS+=-mfpu=neon -DHPTT_ARCH_ARM
arm2: all
ibm2: CXX_FLAGS+=-mcpu=power7 -DHPTT_ARCH_IBM -maltivec -mabi=altivec
ibm2: all
scalar2: all

SRC=$(wildcard ./src/*.cpp)
OBJ=$(SRC:.cpp=.o)
SHARED_OBJ=$(SRC:.cpp=.so)

all: lib/libhptt.so lib/libhptt.a

lib/libhptt.so: ${SHARED_OBJ}
	mkdir -p lib
	${CXX} ${SHARED_OBJ} ${CXX_FLAGS} -o lib/libhptt.so -shared

lib/libhptt.a: ${OBJ}
	mkdir -p lib
	ar rvs lib/libhptt.a ${OBJ}

%.o: %.cpp
	${CXX} ${CXX_FLAGS} ${INCLUDE_PATH} -c $< -o $@

%.so: %.cpp
	${CXX} ${CXX_FLAGS} -fPIC ${INCLUDE_PATH} -c $< -o $@

doc:
	doxygen

clean:
	rm -rf ${OBJ} ${SHARED_OBJ} lib/libhptt.so lib/libhptt.a
