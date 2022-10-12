cmake_minimum_required(VERSION 3.23)
project(my_hf)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
        BTAS
        GIT_REPOSITORY      https://github.com/ValeevGroup/btas.git
)
FetchContent_MakeAvailable(BTAS)
FetchContent_GetProperties(BTAS
        SOURCE_DIR BTAS_SOURCE_DIR
        BINARY_DIR BTAS_BINARY_DIR
        )
#add_library(BTAS)
message(STATUS "BTAS: ${BTAS}")


find_package(Libint2 REQUIRED)


add_executable(my_hf
        my_hf.cpp util.cpp one_body_ints.cpp scf_helper.cpp build_fock.cpp mp2.cpp ccsd.cpp so_integrals.cpp)

target_link_libraries(my_hf BTAS)
target_link_libraries(my_hf Libint2::cxx)