cmake_minimum_required(VERSION 3.10)
project(SuperResONNX)

set(CMAKE_CXX_STANDARD 17)

# Para que find_package busque módulos CUDA de OpenCV también
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui cudaimgproc cudawarping)

include_directories(${OpenCV_INCLUDE_DIRS})

# Ruta a ONNX Runtime (asegúrate de pasarla con -DONNXRUNTIME_DIR=...)
if(NOT DEFINED ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Debe definir ONNXRUNTIME_DIR al ejecutar cmake")
endif()

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(principal principal.cpp)

target_link_libraries(principal ${OpenCV_LIBS} onnxruntime)
