cmake_minimum_required(VERSION 3.9)

project(ONNX2TensorRT)

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(TARGET_NAME ONNX2TensorRT)


IF(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE DEBUG)
ENDIF()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(/usr/local/cuda/include)
include_directories(${PROJECT_SOURCE_DIR}/include/include)
include_directories(${PROJECT_SOURCE_DIR}/include/samples)


link_directories(/home/zwgd/TensorRT-8.5.1.7/lib)

add_executable(ONNX2TensorRT 
    pytorch2ONNX2TensorRT.cpp 
)

# target_link_libraries() 是用来为特定的目标指定链接库的，因此必须在定义该目标之后调用它
target_link_libraries(${TARGET_NAME} nvonnxparser nvinfer)


