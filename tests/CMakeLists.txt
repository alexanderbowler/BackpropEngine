add_executable(all_tests.exe
    tensor_tests.cpp
    function_tests.cpp
    test_helpers.hpp
)

target_compile_definitions(all_tests.exe PRIVATE UNIT_TEST)
target_link_libraries(all_tests.exe PRIVATE gtest_main tensor_test_library)