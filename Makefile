CXX_FLAGS=-O3 -qopenmp -xhost -std=c++14 -fPIC

SRC=$(wildcard ./src/*.cpp)
OBJ=$(SRC:.cpp=.o)

all: ${OBJ}
	${CXX} ${OBJ} ${CXX_FLAGS} -o lib/libhptt.so -shared

%.o: %.cpp
	${CXX} ${CXX_FLAGS} ${INCLUDE_PATH} -c $< -o $@

clean:
	rm -rf src/*.o lib/libhptt.so
