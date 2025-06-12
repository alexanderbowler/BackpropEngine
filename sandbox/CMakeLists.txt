add_executable(main.exe main.cpp)

target_link_libraries(main.exe PRIVATE tensor)
target_include_directories(main.exe PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)