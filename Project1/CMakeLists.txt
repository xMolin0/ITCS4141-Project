cmake_minimum_required(VERSION 3.6)
project( OpenCVExample )

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(OpenCV_DIR "C:/Users/xavie/Downloads/opencv/build/x64/vc16/lib")
find_package( OpenCV REQUIRED )

add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )

add_executable( FilteringOpenCV FilteringOpenCV.cpp )
target_link_libraries( FilteringOpenCV ${OpenCV_LIBS} )

add_executable( Filtering Filtering.cpp )
target_link_libraries( Filtering ${OpenCV_LIBS} )