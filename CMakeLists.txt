# require CMake 2.8 or greater
cmake_minimum_required(VERSION 2.8)

# declare JetsonNano_Recognition project
project(JetsonNano_Recognition)

# import jetson-inference and jetson-utils packages.
# not that if you didn't do "sudo make install"
# while building jetson-inference, this will error.
find_package(jetson-utils)
find_package(jetson-inference)

# CUDA and Qt4 are required
find_package(CUDA)
find_package(Qt4)

# setup Qt4 for build 
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

# compile the JetsonNano_Recognition program
cuda_add_executable(JetsonNano_Recognition JetsonNano_Recognition.cpp)

# link JetsonNano_Recognition to jetson-inference library
target_link_libraries(JetsonNano_Recognition jetson-inference)