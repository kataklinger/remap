cmake_minimum_required (VERSION 3.18)

project ("remap")

find_package(libpng CONFIG REQUIRED)

add_executable (remap "main.cpp" "cpl.hpp" "mrl.hpp" "kpr.hpp" "kpe_v1.hpp" "kpe_v2.hpp" "pngu.hpp")

target_link_libraries(remap PRIVATE png)
target_compile_features(remap PUBLIC cxx_std_20)
