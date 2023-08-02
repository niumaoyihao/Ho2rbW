cmake_minimum_required( VERSION 2.8 )
project( hf )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
set(OpenCV_DIR "E:/niumaoyihao/VScode-workspace/opencv/opencv4.5.2")

find_package(OpenCV REQUIRED)

#find_package(OpenCV 3.4.5 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( hf hf.cpp )

target_link_libraries( hf ${OpenCV_LIBS} )


