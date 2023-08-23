cmake_minimum_required(VERSION 3.9)

project(CppShizhan)

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(TARGET_NAME CppShizhan)

IF(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE DEBUG)
ENDIF()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(/usr/local/cuda/include)
include_directories(${PROJECT_SOURCE_DIR}/include/include)
include_directories(${PROJECT_SOURCE_DIR}/include/samples)

link_directories(/home/zwgd/TensorRT-8.5.1.7/lib)

add_executable(CppShizhan 
        Cppshizhan.cpp
)

# Set linker flags
set(CUDA_LINKER_FLAGS "-L/usr/local/cuda-11.6/targets/x86_64-linux/lib")
set(NVINFER_LINKER_FLAGS "-lnvinfer")

# Append CUDA and NVINFER linker flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CUDA_LINKER_FLAGS} ${NVINFER_LINKER_FLAGS}")

# Link libraries
target_link_libraries(${TARGET_NAME} nvonnxparser nvinfer cudart)

