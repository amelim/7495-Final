cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(CV-3)


find_package(PCL 1.6 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (active src/active.cpp)
target_link_libraries (active ${PCL_LIBRARIES})
target_link_libraries (active ${OpenCV_LIBRARIES})
