cmake_minimum_required(VERSION 3.23)
project(my_hf)

set(CMAKE_CXX_STANDARD 11)

find_package(Libint2 REQUIRED)

add_executable(my_hf
        my_hf.cpp util.cpp one_body_ints.cpp scf_helper.cpp build_fock.cpp mp2.cpp)

target_link_libraries(my_hf Libint2::cxx)