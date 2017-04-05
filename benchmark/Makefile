CXX_FLAGS=-O3 -qopenmp -xhost -std=c++14

SRC=benchmark.cpp
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
