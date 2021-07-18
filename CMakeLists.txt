cmake_minimum_required (VERSION 3.18)

project ("remap")

find_package(libpng CONFIG REQUIRED)

add_executable (remap
	"src/all.hpp"
	"src/cpl.hpp"
	"src/mrl.hpp"
	"src/cdt.hpp"
	"src/kpr.hpp"
	"src/kpe.hpp"
	"src/kpe_v2.hpp"
	"src/kpm.hpp"
	"src/ctr.hpp"
	"src/cte.hpp"
	"src/mod.hpp"
	"src/fgm.hpp"
	"src/frc.hpp"
	"src/fgs.hpp"
	"src/ifd.hpp"
	"src/fde.hpp"
	"src/aws.hpp"
	"src/fdf.hpp"
	"src/arf.hpp"
	"src/icd.hpp"
	"src/nic.hpp"
	"src/mpb.hpp"
	"src/pngu.hpp"
	"src/main.cpp" "src/sid.hpp")

target_link_libraries(remap PRIVATE png)
target_compile_features(remap PUBLIC cxx_std_20)
