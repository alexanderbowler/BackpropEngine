# Production library without tests
add_library(tensor STATIC tensor.cpp function.cpp)

target_include_directories(tensor PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Test library that includes gtest for tests
add_library(tensor_test_library STATIC 
    tensor.cpp function.cpp constantRegistry.cpp
)

target_compile_definitions(tensor_test_library PRIVATE UNIT_TEST)
target_include_directories(tensor_test_library PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
