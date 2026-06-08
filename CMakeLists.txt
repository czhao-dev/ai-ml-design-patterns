cmake_minimum_required(VERSION 3.22)

project(tensor_graph_inference_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(tge INTERFACE)
target_include_directories(tge INTERFACE include)
target_compile_features(tge INTERFACE cxx_std_17)

add_executable(compile_model tools/compile_model.cpp)
target_link_libraries(compile_model PRIVATE tge)

add_executable(infer src/infer_main.cpp)
target_link_libraries(infer PRIVATE tge)

enable_testing()
foreach(test_name test_ops test_arena_planner test_end_to_end test_zero_alloc)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE tge)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
