cmake_minimum_required(VERSION 3.18)
project(pyai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(WITH_CUDA "Build with CUDA GPU support" OFF)
option(WITH_VULKAN "Build with Vulkan inference support" OFF)

if(WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
  add_compile_definitions(WITH_CUDA)
  message(STATUS "CUDA enabled — architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "CUDA disabled — CPU-only build")
endif()

if(WITH_VULKAN)
  find_package(Vulkan REQUIRED)
  add_compile_definitions(WITH_VULKAN)
  message(STATUS "Vulkan inference enabled")
else()
  message(STATUS "Vulkan inference disabled")
endif()

if(MSVC)
  add_compile_options(/O2 /W3 /MP /utf-8)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
  add_compile_options(-O3 -march=native -Wall -Wextra -Wno-unused-parameter)
  if(UNIX AND NOT APPLE)
    add_link_options(-pthread)
  endif()
endif()

set(SOURCES
  src/main.cpp
  src/vocab.cpp
  src/model.cpp
  src/trainer.cpp
  src/generator.cpp
  src/tokenizer.cpp
  src/settings.cpp
  src/downloader.cpp
  src/utils.cpp
)

if(WITH_VULKAN)
  list(APPEND SOURCES vulkan/vulkan_ops.cpp)
  set(VULKAN_SHADER ${CMAKE_BINARY_DIR}/head_projection.spv)
  add_custom_command(
    OUTPUT ${VULKAN_SHADER}
    COMMAND glslc ${CMAKE_SOURCE_DIR}/vulkan/head_projection.comp -o ${VULKAN_SHADER}
    DEPENDS ${CMAKE_SOURCE_DIR}/vulkan/head_projection.comp
    VERBATIM)
  add_custom_target(vulkan_shaders DEPENDS ${VULKAN_SHADER})
endif()

if(WITH_CUDA)
  list(APPEND SOURCES cuda/cuda_ops.cu)
endif()

add_executable(pyai ${SOURCES})
if(WITH_VULKAN)
  add_dependencies(pyai vulkan_shaders)
  target_compile_definitions(pyai PRIVATE VULKAN_SHADER_PATH="${VULKAN_SHADER}")
  target_link_libraries(pyai PRIVATE Vulkan::Vulkan)
endif()

target_include_directories(pyai PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/cuda
  ${CMAKE_SOURCE_DIR}/include
)

if(WITH_CUDA)
  target_include_directories(pyai PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(pyai PRIVATE CUDA::cublas CUDA::curand)
  set_target_properties(pyai PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

find_package(ZLIB REQUIRED)
target_link_libraries(pyai PRIVATE ZLIB::ZLIB)

find_package(Threads REQUIRED)
target_link_libraries(pyai PRIVATE Threads::Threads)

if(WIN32)
  target_link_libraries(pyai PRIVATE ws2_32 winhttp)
endif()

install(TARGETS pyai DESTINATION bin)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
