# CMakeList.txt : CMake project for CMakeProject4, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Vec256Test")
if(MSVC)
set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# so the build can find the generated header files

list(APPEND CPU_INCLUDE ${CMAKE_CURRENT_BINARY_DIR})
include(CheckCSourceCompiles)
include(CheckCXXCompilerFlag)

include(FindVsx.cmake)
include(FindAVX.cmake)

add_subdirectory(c10)
link_directories(${CMAKE_BINARY_DIR}/c10)
list(APPEND DEPENDENCY_LIBS c10)
if(MSVC)
    set(OPT_FLAG "/fp:strict /bigobj ") 
else(MSVC)
    set(OPT_FLAG "-O3 -pthread   -fmax-errors=4")
    if("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
      set(OPT_FLAG " ")
    endif() 
endif(MSVC)

list(APPEND CPU_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND CPU_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ATen/cpu)

list(APPEND CPU_CAPABILITY_NAMES "DEFAULT")
list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}")

if(CXX_AVX_FOUND)
    list(APPEND CPU_CAPABILITY_NAMES "AVX")
    if(MSVC)
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX")
    else(MSVC)
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -mavx")
    endif(MSVC)
endif(CXX_AVX_FOUND)

if(CXX_AVX2_FOUND)
    # Some versions of GCC pessimistically split unaligned load and store
    # instructions when using the default tuning. This is a bad choice on
    # new Intel and AMD processors so we disable it when compiling with AVX2.
    # See https://stackoverflow.com/questions/52626726/why-doesnt-gcc-resolve-mm256-loadu-pd-as-single-vmovupd#tab-top
    check_cxx_compiler_flag("-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store" COMPILER_SUPPORTS_NO_AVX256_SPLIT)
    if(COMPILER_SUPPORTS_NO_AVX256_SPLIT)
      set(CPU_NO_AVX256_SPLIT_FLAGS "-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store")
    endif(COMPILER_SUPPORTS_NO_AVX256_SPLIT)

    list(APPEND CPU_CAPABILITY_NAMES "AVX2")
    if(MSVC)
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX2")
    else(MSVC)
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -mavx2 -mfma ${CPU_NO_AVX256_SPLIT_FLAGS}")
    endif(MSVC)
endif(CXX_AVX2_FOUND)

if(CXX_VSX_FOUND)
    LIST(APPEND CPU_CAPABILITY_NAMES "VSX")
    LIST(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}  ${CXX_VSX_FLAGS}") 
endif(CXX_VSX_FOUND)

list(LENGTH CPU_CAPABILITY_NAMES NUM_CPU_CAPABILITY_NAMES)
math(EXPR NUM_CPU_CAPABILITY_NAMES "${NUM_CPU_CAPABILITY_NAMES}-1")

# Preserve values for the main build
set(__aten_sleef_build_shared_libs ${BUILD_SHARED_LIBS})
set(__aten_sleef_build_tests ${BUILD_TESTS})

# Unset our restrictive C++ flags here and reset them later.
# Remove this once we use proper target_compile_options.
set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(OLD_CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}) 
set(CMAKE_CXX_FLAGS)

# Bump up optimization level for sleef to -O1, since at -O0 the compiler
# excessively spills intermediate vector registers to the stack
# and makes things run impossibly slowly
set(OLD_CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
if(${CMAKE_C_FLAGS_DEBUG} MATCHES "-O0")
    string(REGEX REPLACE "-O0" "-O1" CMAKE_C_FLAGS_DEBUG ${OLD_CMAKE_C_FLAGS_DEBUG})
else()
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O1")
endif()
  IF(${CMAKE_C_FLAGS_DEBUG} MATCHES "/RTC1")
    string(REGEX REPLACE "/RTC1" "" CMAKE_C_FLAGS_DEBUG  ${OLD_CMAKE_C_FLAGS_DEBUG} )
   
  ENDIF()
if(NOT USE_SYSTEM_SLEEF AND NOT MSVC)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build sleef static" FORCE)
    set(BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
    set(BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)
    set(OLD_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
    if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND
        CMAKE_C_COMPILER_VERSION VERSION_GREATER 6.9 AND CMAKE_C_COMPILER_VERSION VERSION_LESS 8)
      set(GCC_7 True)
    else()
      set(GCC_7 False)
    endif()
    if(GCC_7)
      set(CMAKE_BUILD_TYPE Release)  # Always build Sleef as a Release build to work around a gcc-7 bug
    endif()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/sleef ${CMAKE_BINARY_DIR}/sleef)
    if(GCC_7)
      set(CMAKE_BUILD_TYPE ${OLD_CMAKE_BUILD_TYPE})
    endif()
    set_property(TARGET sleef PROPERTY FOLDER "dependencies")
    list(APPEND ATen_THIRD_PARTY_INCLUDE ${CMAKE_BINARY_DIR}/include)
    link_directories(${CMAKE_BINARY_DIR}/sleef/lib)
    list(APPEND DEPENDENCY_LIBS sleef)
endif()


set(CMAKE_C_FLAGS_DEBUG ${OLD_CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})

# Set these back. TODO: Use SLEEF_ to pass these instead
set(BUILD_SHARED_LIBS ${__aten_sleef_build_shared_libs} CACHE BOOL "Build shared libs" FORCE)
set(BUILD_TESTS ${__aten_sleef_build_tests} CACHE BOOL "Build tests" FORCE)

if(NOT USE_SYSTEM_GTEST)
  # Preserve build options.
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

  # We will build gtest as static libs and embed it directly into the binary.
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)

  # For gtest, we will simply embed it into our test binaries, so we won't
  # need to install it.
  set(INSTALL_GTEST OFF CACHE BOOL "Install gtest." FORCE)
  set(BUILD_GMOCK ON CACHE BOOL "Build gmock." FORCE)
  # For Windows, we will check the runtime used is correctly passed in.
  if(NOT CAFFE2_USE_MSVC_STATIC_RUNTIME)
      set(gtest_force_shared_crt ON CACHE BOOL "force shared crt on gtest" FORCE)
  endif()
  # We need to replace googletest cmake scripts too.
  # Otherwise, it will sometimes break the build.
  # To make the git clean after the build, we make a backup first.
  if(MSVC AND MSVC_Z7_OVERRIDE)
    execute_process(
      COMMAND ${CMAKE_COMMAND}
              "-DFILENAME=${CMAKE_CURRENT_LIST_DIR}/third_party/googletest/googletest/cmake/internal_utils.cmake"
              "-DBACKUP=${CMAKE_CURRENT_LIST_DIR}/third_party/googletest/googletest/cmake/internal_utils.cmake.bak"
              "-DREVERT=0"
              "-P"
              "${CMAKE_CURRENT_LIST_DIR}/GoogleTestPatch.cmake"
      RESULT_VARIABLE _exitcode)
    if(NOT _exitcode EQUAL 0)
      message(WARNING "Patching failed for Google Test. The build may fail.")
    endif()
  endif()

  # Add googletest subdirectory but make sure our INCLUDE_DIRECTORIES
  # don't bleed into it. This is because libraries installed into the root conda
  # env (e.g. MKL) add a global /opt/conda/include directory, and if there's
  # gtest installed in conda, the third_party/googletest/**.cc source files
  # would try to include headers from /opt/conda/include/gtest/**.h instead of
  # its own. Once we have proper target-based include directories,
  # this shouldn't be necessary anymore.
  get_property(INC_DIR_temp DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
  set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES "")
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/googletest)
  set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES ${INC_DIR_temp})

  include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/third_party/googletest/googletest/include)
  include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/third_party/googletest/googlemock/include)

endif()

  # Recover build options.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)

 
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")# -Wall -fstrict-aliasing  -fsanitize=undefined -fsanitize=float-cast-overflow")
if(MSVC)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O3 ")
else()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -fsanitize=undefined -fsanitize=address  -fno-sanitize-recover=all -std=gnu++14  ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall  -O3  -std=gnu++14 ")
endif()
message("&&&& ${CMAKE_CXX_FLAGS} &&")


  foreach(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
        set(test_name vec256_test_all_types)
        list(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY) 
        list(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)  
        separate_arguments(FLAGS UNIX_COMMAND "${FLAGS}")
        add_executable(${test_name}_${CPU_CAPABILITY} ${CMAKE_CURRENT_SOURCE_DIR}/ATen/test/vec256_test_all_types.cpp)
        target_link_libraries(${test_name}_${CPU_CAPABILITY} ${DEPENDENCY_LIBS} gtest gtest_main)
        target_include_directories(${test_name}_${CPU_CAPABILITY} PRIVATE $<INSTALL_INTERFACE:include>)
        target_include_directories(${test_name}_${CPU_CAPABILITY} PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)
        target_include_directories(${test_name}_${CPU_CAPABILITY} PRIVATE ${CPU_INCLUDE})
        target_compile_definitions(${test_name}_${CPU_CAPABILITY} PRIVATE CPU_CAPABILITY=${CPU_CAPABILITY}  CPU_CAPABILITY_${CPU_CAPABILITY})
        target_compile_options(${test_name}_${CPU_CAPABILITY} PRIVATE  ${FLAGS})
        if(NOT MSVC)
              target_compile_options(${test_name}_${CPU_CAPABILITY} PRIVATE -Wno-ignored-qualifiers) 
        endif(NOT MSVC)
        add_test(NAME ${test_name}_${CPU_CAPABILITY} COMMAND $<TARGET_FILE:${test_name}_${CPU_CAPABILITY}>)
  endforeach()
 
