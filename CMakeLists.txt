cmake_minimum_required(VERSION 3.16)
project(BowlerBackprop VERSION 0.1 LANGUAGES CXX)

#Fetch GTEST framework
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)
set(GTEST_IMPORTED_TARGETS_DEFINED TRUE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

include_directories(include/backprop)

add_subdirectory(src/tensor)
add_subdirectory(sandbox)
add_subdirectory(tests)