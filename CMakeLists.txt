cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(CV-3)


find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(icp src/segmentation.cpp)
target_link_libraries(icp ${PCL_LIBRARIES})

add_executable(segment src/grow.cpp)
target_link_libraries(segment ${PCL_LIBRARIES})
