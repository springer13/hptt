CXX_FLAGS=-O3 -qopenmp -xhost -std=c++14

SRC=$(wildcard *.cc)
OBJ=$(SRC:.cc=.o)
TYPE=cc

INCLUDE_PATH=-I/home/ps072922/projects/ttc-c/include/
LIB_PATH=-L/home/ps072922/projects/ttc-c/build/src/
LIBS=-lttc_c

intel: 
	${MAKE} clean 
	${MAKE} intel2
#intel2: CXX=icpc
#intel2: CXX_FLAGS += -qopenmp -xhost -restrict 
intel2: all


all: ${OBJ}
	${CXX} ${OBJ} ${LIB_PATH} ${LIBS} ${CXX_FLAGS} -o transpose.exe

%.o: %.${TYPE}
	${CXX} ${CXX_FLAGS} ${INCLUDE_PATH} -c $< -o $@

clean:
	rm -rf *.o transpose.exe
