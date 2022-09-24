cmake_minimum_required(VERSION 3.23)
project(my_hf)

set(CMAKE_CXX_STANDARD 11)

add_executable(my_hf
        my_hf.cpp util.cpp one_body_ints.cpp scf_helper.cpp build_fock.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC "/Users/kshitijsurjuse/Desktop/libraries/eigen-3.4.0"
        "/opt/homebrew/opt/boost/include"
        "/Users/kshitijsurjuse/Desktop/project1/install/libint2/include")
target_link_directories(${PROJECT_NAME} PUBLIC "/Users/kshitijsurjuse/Desktop/libraries/eigen-3.4.0"
        "/opt/homebrew/opt/boost/lib"
        "/Users/kshitijsurjuse/Desktop/project1/install/libint2/lib")

target_link_libraries(my_hf int2)