cmake_minimum_required(VERSION 3.20.0)

project(trainingModel LANGUAGES CXX C)

set(NNMODEL_LIB_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(NNMODEL_INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/include)

include_directories(${NNMODEL_INCLUDE_DIRECTORY})
link_directories(${NNMODEL_LIB_DIRECTORY})

set(SOURCES ${NNMODEL_LIB_DIRECTORY}/neuralNetModels.c main.c)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(main ${SOURCES})

