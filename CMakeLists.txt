cmake_minimum_required(VERSION 3.12)
enable_language(CUDA)
project(Matrix_Multiplication_with_CUDA)

include_directories(include)

set(HEADERS include/utils.hpp)

add_executable(getDeviceProps src/getDeviceProps.cu)
add_executable(matrixMulKernel ${HEADERS} src/matrixMulKernel.cu)
add_executable(matrixMulKernelShared ${HEADERS} src/matrixMulKernelShared.cu)

set_property(TARGET getDeviceProps PROPERTY CUDA_ARCHITECTURES 80;86)
set_property(TARGET matrixMulKernel PROPERTY CUDA_ARCHITECTURES 80;86)
set_property(TARGET matrixMulKernelShared PROPERTY CUDA_ARCHITECTURES 80;86)

configure_file(${CMAKE_SOURCE_DIR}/scripts/test.sh ${CMAKE_BINARY_DIR}/test.sh COPYONLY)

add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/Makefile
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)