enable_testing()

set(COMMON_CXX_FLAGS "-O3 -std=c++11")

# AppleClang does not support OpenMP, therefore not MATCHES but STREQUAL:
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS} -fopenmp")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS} -qopenmp -xhost")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # PPC does not support -march=
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc|ppc64|ppc64le" OR CMAKE_OSX_ARCHITECTURES MATCHES "ppc|ppc64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS} -fopenmp -mtune=native")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS} -fopenmp -march=native")
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS}")
endif()

set(TEST_PATH ${CMAKE_CURRENT_BINARY_DIR})

include_directories(
  ../include
  ../benchmark
)

set(testframework_SRCS
  testframework.cpp
  ../benchmark/reference.cpp
)

add_executable(testframework ${testframework_SRCS})
target_link_libraries(hptt INTERFACE "-L../lib/")

add_test(testframework ${TEST_PATH}/testframework)
