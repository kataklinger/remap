cmake_minimum_required (VERSION 3.18)

project ("remap")

find_package(libpng CONFIG REQUIRED)

add_executable (remap
	"src/all.hpp"
	"src/cpl.hpp"
	"src/mrl.hpp"
	"src/kpr.hpp"
	"src/kpe_v1.hpp"
	"src/kpe_v2.hpp"
	"src/kpm.hpp"
	"src/cte_v1.hpp"
	"src/pngu.hpp"
	"src/main.cpp")

target_link_libraries(remap PRIVATE png)
target_compile_features(remap PUBLIC cxx_std_20)
