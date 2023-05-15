
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project (HPTT VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON)

option(ENABLE_IBM OFF)

if(NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif()

include(CheckCXXCompilerFlag)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
  set(ENABLE_IBM ON)
endif()


if(DEFINED MARCH_FLAGS)
  set(HPTT_ARCH_FLAGS ${MARCH_FLAGS})
else()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(HPTT_ARCH_FLAGS -xhost)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(NOT ENABLE_IBM)
      set(HPTT_ARCH_FLAGS -march=native) # -mtune=native
    endif() 
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(HPTT_ARCH_FLAGS -march=native)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
    set(HPTT_ARCH_FLAGS -silent -w -Mnovect)
  # elseif(CMAKE_CXX_COMPILER_ID STREQUAL "XL")
  #   set(HPTT_ARCH_FLAGS -qsmp=omp)
  #elseif(CMAKE_CXX_COMPILER_ID STREQUAL "ARMClang")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(HPTT_ARCH_FLAGS -mcpu=native)
  endif()
endif()

check_cxx_compiler_flag("${HPTT_ARCH_FLAGS}" __COMPILER_SUPPORTS_MARCH)
if(__COMPILER_SUPPORTS_MARCH)
  set(HPTT_CXX_FLAGS "${HPTT_ARCH_FLAGS}")
endif()

if(ENABLE_AVX)
  set(HPTT_CXX_FLAGS ${HPTT_CXX_FLAGS} -mavx -DHPTT_ARCH_AVX)
elseif(ENABLE_ARM)
  set(HPTT_CXX_FLAGS ${HPTT_CXX_FLAGS} -mfpu=neon -DHPTT_ARCH_ARM)
elseif(ENABLE_IBM)
  set(HPTT_CXX_FLAGS ${HPTT_CXX_FLAGS} -mtune=native -DHPTT_ARCH_IBM -maltivec -mabi=altivec)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "8.2")
  set(HPTT_CXX_FLAGS ${HPTT_CXX_FLAGS} -O2)
endif()

set(HPTT_SRCS src/hptt.cpp src/plan.cpp src/transpose.cpp src/utils.cpp)

add_library(hptt ${HPTT_SRCS})

add_library(hptt::hptt ALIAS hptt)

target_include_directories(hptt 
    PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_options(hptt PRIVATE ${HPTT_CXX_FLAGS})

if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(hptt PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(hptt PROPERTIES EXPORT_NAME hptt)

# Install

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/hptt)

install(TARGETS hptt
    EXPORT hptt-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)


install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hptt)

#Export the targets to a script
install(EXPORT hptt-targets
    FILE
        hptt-targets.cmake
    NAMESPACE
        hptt::
    DESTINATION
        ${INSTALL_CONFIGDIR}
)

#Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/hptt-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/hptt-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/hptt-config.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

#Install the config, configversion and custom find modules
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hptt-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/hptt-config-version.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)


export(EXPORT hptt-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/hptt-targets.cmake
    NAMESPACE hptt::)

#Register package in user's package registry
export(PACKAGE hptt)
