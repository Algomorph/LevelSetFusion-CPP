#  ================================================================
#  Created by Gregory Kramida on 10/10/18.
#  Copyright (c) 2018 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================
#============================================================================================

# TODO: up the required CMake version to 3.12.4 and replace fancy python utilities with `find_package(Python3 REQUIRED)`

# Preliminaries for CMake
cmake_minimum_required(VERSION 3.5.1)
# Put our local cmake find scripts at the beginning of the CMake
# module search path.
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#=================================================================
# PYTHON option & interpreter detection

option(LSFusion_PYTHON3_VERSION "Python3 version" "")

include("PythonUtilities")

# TODO: not sure if this line is needed for anything -- to be removed
#Python_ADDITIONAL_VERSIONS is only used inside the find_python_interpreter on Windows
#set(Python_ADDITIONAL_VERSIONS ${PYTHON3_VERSION_MAJOR}.${PYTHON3_VERSION_MINOR})

find_python_interpreter(${LSFusion_PYTHON3_VERSION} 3.4 PYTHON3_LIBRARY PYTHON3_INCLUDE_DIR
            			PYTHON3INTERP_FOUND PYTHON3_EXECUTABLE PYTHON3_VERSION_STRING
            			PYTHON3_VERSION_MAJOR PYTHON3_VERSION_MINOR PYTHON3_VERSION_PATCH)

#=================================================================
# Preliminaries for Hunter
# TODO: reset HUNTER_ROOT to default

include("HunterGate")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.125.tar.gz"
    SHA1 "458d4ffb3e62c86322943baf5b94a3a21a6b9397"
    LOCAL
)

#=================================================================
# Project & main target naming
project(LevelSetFusionOptimization LANGUAGES CXX)
# Name of the target and python module
set(MODULE_NAME level_set_fusion_optimization)
# TODO: replace with toolchain file, here and in `cmake/Hunter/config.cmake`?
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#=================================================================
# Compiler options

# Preliminaries for Compiler
# C++11 standard simplifies things in a lot of obvious ways,
# it also allows us to use Eigen in multithreaded code w/o special initialization
set(CMAKE_CXX_STANDARD 11)

if ( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
	# -Wno-ignored-attributes -- avoiding the ignored attribute warning when mapping tensors in Eigen 3.3.7
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-deprecated-declarations -fms-extensions -Wno-ignored-attributes ")
    if(NOT HUNTER_ENABLED)
    	set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-int-in-bool-context -Wno-ignored-attributes")
    endif()
endif()
if ( MSVC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W4")
endif()
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")

#=================================================================
# PYTHON library detection & final configuration

find_python(${LSFusion_PYTHON3_VERSION} 3.4 PYTHON3_LIBRARY PYTHON3_INCLUDE_DIR
            ${PYTHON3INTERP_FOUND} ${PYTHON3_EXECUTABLE} ${PYTHON3_VERSION_STRING}
            ${PYTHON3_VERSION_MAJOR} ${PYTHON3_VERSION_MINOR} ${PYTHON3_VERSION_PATCH}
            PYTHON3LIBS_FOUND
            PYTHON3LIBS_VERSION_STRING PYTHON3_LIBRARIES PYTHON3_LIBRARY
            PYTHON3_DEBUG_LIBRARIES PYTHON3_LIBRARY_DEBUG PYTHON3_INCLUDE_PATH
            PYTHON3_INCLUDE_DIR PYTHON3_INCLUDE_DIR2 PYTHON3_PACKAGES_PATH
            PYTHON3_NUMPY_INCLUDE_DIRS PYTHON3_NUMPY_VERSION)

set(PYTHON3_INCLUDE_DIRS ${PYTHON3_INCLUDE_DIR} ${PYTHON3_INCLUDE_DIR2} ${PYTHON3_NUMPY_INCLUDE_DIRS})

set(ARCHIVE_OUTPUT_NAME ${MODULE_NAME}_py3)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Python DEFAULT_MSG PYTHON3_INCLUDE_DIRS PYTHON3_NUMPY_INCLUDE_DIRS PYTHON3_LIBRARIES PYTHON3_EXECUTABLE PYTHON3_PACKAGES_PATH)
if(NOT Python_FOUND)
    message(SEND_ERROR "Not all requred components of Numpy/Python found.")
endif()

# Find true site-packages site on this machine, since find_package(PythonLibs) doesn't seem to get the correct one
# (Adapted from OpenCV3)
execute_process(COMMAND ${PYTHON3_EXECUTABLE} -c
        "from distutils.sysconfig import get_python_lib;\
  print(get_python_lib(plat_specific=True, prefix=''))"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

set(MODULE_INSTALL_PREFIX "${MODULE_INSTALL_PREFIX}" CACHE STRING
        "Output directory for Python modules.")

set(__python_ext_suffix_var "EXT_SUFFIX")
if({PYTHON_DESIRED_VERSION} STREQUAL "2.X")
    set(__python_ext_suffix_var "SO")
endif()

# get proper extension for python binary shared object on this platform
execute_process(COMMAND ${PYTHON3_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('${__python_ext_suffix_var}'))"
        RESULT_VARIABLE PYTHON_${MODULE_NAME}_PY_PROCESS
        OUTPUT_VARIABLE ${MODULE_NAME}_PY_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT ${PYTHON_${PROJECT_NAME}_PY_PROCESS} EQUAL 0)
    set(${PROJECT_NAME}_PY_SUFFIX ".so")
endif()

#=================================================================
#Hunter package discovery
hunter_add_package(Boost COMPONENTS test python)
hunter_add_package(Eigen)

#==================== FIND PACKAGES ==============================
include_directories(include)

# *** Boost (Boost python version based on python option / availability)
if(NOT HUNTER_ENABLED)
	if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
	    # Provisions for typical Boost compiled on Windows
	    # Most commonly, Boost libraries are compiled statically on windows (change as necesssary)
	    set(Boost_USE_STATIC_LIBS TRUE)
	    set(Boost_USE_STATIC_RUNTIME OFF)
	    set(Boost_USE_MULTITHREADED ON)
	    set(Boost_USE_DEBUG_RUNTIME ON)
	    set(Boost_USE_DEBUG_PYTHON OFF)
	endif()
endif()

set(BOOST_COMPONENTS unit_test_framework)
set(BOOST_ALTERNATIVE_COMPONENTS unit_test_framework)

list(APPEND BOOST_COMPONENTS python${PYTHON3_VERSION_MAJOR}${PYTHON3_VERSION_MINOR})
list(APPEND BOOST_ALTERNATIVE_COMPONENTS python-py${PYTHON3_VERSION_MAJOR}${PYTHON3_VERSION_MINOR})
set(PYTHON3_EXECUTABLE ${Python3_EXECUTABLE})

find_package(Boost COMPONENTS ${BOOST_COMPONENTS} QUIET)
if(NOT Boost_FOUND)
	message(STATUS "Trying alternative Boost.Python component name, python-py<version>...")
	find_package(Boost COMPONENTS ${BOOST_ALTERNATIVE_COMPONENTS} REQUIRED)
endif()

# *** Everything else
find_package(Eigen3 3.3.4 REQUIRED)
find_package(OpenMP REQUIRED)
if ( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
	find_package(GCCAtomic REQUIRED)
endif()
#=================== DEBUG & VERBOSITY OPTIONS =========================================================================

option(SDF_GENERATION_CONSOLE_PROGRESS_REPORTS "SDF generation will print completion to stdout at fixed progress intervals." OFF)

#=================== HEADER CONFIGURATION ==============================================================================

configure_file(include/lsf_config.h.in include/lsf_config.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

#=================== TARGET DEFININTIONS ===============================================================================

# main library
add_library(${MODULE_NAME} SHARED

			src/module.cpp

			### python export: all code for porting everything from the other namespaces (as nescessary) to Python
			src/python_export/conversion_tests.hpp
			src/python_export/conversion_tests.cpp
			src/python_export/eigen_numpy.hpp
			src/python_export/eigen_numpy_matrix.cpp
			src/python_export/eigen_numpy_tensor.cpp
			src/python_export/eigen_numpy_list.cpp
			src/python_export/hierarchical_optimizer.hpp
			src/python_export/hierarchical_optimizer.cpp
			src/python_export/telemetry.hpp
			src/python_export/telemetry.cpp
			src/python_export/math.hpp
			src/python_export/math.cpp
			src/python_export/numpy_conversions_shared.hpp
			src/python_export/numpy_conversions_shared.cpp
			src/python_export/slavcheva_optimizer.hpp
			src/python_export/slavcheva_optimizer.cpp
			src/python_export/tsdf.hpp
			src/python_export/tsdf.cpp

			#### nonrigid_optimization: namespace for algorithms performing non-rigid optimization on TSDF grids #######
			# TSDF: voxel grid containing truncated signed distances to nearest surface in the volume (F is "field")
			# Level set == different name for "approximately" the same concept
			# for more details, see KillingFusion & SobolevFusion publications by Slavcheva et al.,
			# VolumeDeform by Innmann et al. and DynamicFusion by Newcombe et al.

			src/nonrigid_optimization/field_warping.hpp
			src/nonrigid_optimization/field_warping.tpp
			src/nonrigid_optimization/field_warping.cpp

			### nonrigid optimization, slavcheva: KillingFusion/Sobolev fusion
			src/nonrigid_optimization/slavcheva/data_term.hpp
			src/nonrigid_optimization/slavcheva/data_term.cpp
			src/nonrigid_optimization/slavcheva/full_gradient.hpp
			src/nonrigid_optimization/slavcheva/full_gradient.cpp
			src/nonrigid_optimization/slavcheva/smoothing_term.hpp
			src/nonrigid_optimization/slavcheva/smoothing_term.cpp
			src/nonrigid_optimization/slavcheva/optimizer2d.hpp
			src/nonrigid_optimization/slavcheva/optimizer2d.cpp
			src/nonrigid_optimization/slavcheva/sobolev_optimizer2d.hpp
			src/nonrigid_optimization/slavcheva/sobolev_optimizer2d.cpp
			

			### nonrigid optimization, hierarchical: hierarchical SLAM optimizer (active research & development)
			src/nonrigid_optimization/hierarchical/optimizer.hpp
			src/nonrigid_optimization/hierarchical/optimizer.tpp
			src/nonrigid_optimization/hierarchical/optimizer.cpp
			src/nonrigid_optimization/hierarchical/optimizer_with_telemetry.hpp
			src/nonrigid_optimization/hierarchical/optimizer_with_telemetry.tpp
			src/nonrigid_optimization/hierarchical/optimizer_with_telemetry.cpp
			src/nonrigid_optimization/hierarchical/pyramid.hpp
			src/nonrigid_optimization/hierarchical/pyramid.tpp
			src/nonrigid_optimization/hierarchical/pyramid.cpp

			### traversal: algorithms for traversal of data structures
			src/traversal/field_traversal_cpu.hpp
			src/traversal/index_raveling.hpp

			### telemetry: namespace for logging progress and storing up intermediate results for later analysis #######
			src/telemetry/optimization_iteration_data.hpp
			src/telemetry/optimization_iteration_data.tpp
			src/telemetry/optimization_iteration_data.cpp
			src/telemetry/warp_delta_statistics.hpp
			src/telemetry/warp_delta_statistics.cpp
			src/telemetry/tsdf_difference_statistics.hpp
			src/telemetry/tsdf_difference_statistics.cpp
			src/telemetry/convergence_report.hpp
			src/telemetry/convergence_report.cpp

			### math: mathematical utilities
      		src/math/checks.hpp
      		src/math/almost_equal.hpp
      		src/math/almost_equal.tpp
      		src/math/almost_equal.cpp
			src/math/boolean_operations.hpp
			src/math/conics.hpp
			src/math/conics.cpp
			src/math/container_traits.hpp
			src/math/container_traits.tpp
			src/math/container_traits.cpp
			src/math/convolution.hpp
			src/math/convolution.cpp
			src/math/cwise_unary.hpp
			src/math/cwise_unary.tpp
			src/math/cwise_unary.cpp
			src/math/field_like.hpp
			src/math/field_like.tpp
			src/math/field_like.cpp
			src/math/filtered_statistics.hpp
			src/math/filtered_statistics.cpp
			src/math/gradients.hpp
			src/math/gradients.tpp
			src/math/gradients.cpp
			src/math/math_utils.hpp
			src/math/matrix_base.hpp
			src/math/matrix2.hpp
			src/math/multiply.hpp
			src/math/multiply.tpp
			src/math/multiply.cpp
			src/math/nestable_type_metainfo.hpp
			src/math/padding.hpp
			src/math/padding.tpp
			src/math/padding.cpp
			src/math/platform_independence.hpp
			src/math/resampling.hpp
			src/math/resampling.tpp
			src/math/resampling.cpp
			src/math/statistics.hpp
			src/math/statistics.tpp
			src/math/statistics.cpp
			src/math/stacking.hpp
			src/math/stacking.tpp
			src/math/stacking.cpp
			src/math/typedefs.hpp
			src/math/vector_base.hpp
			src/math/vector_operations.hpp
			src/math/vector2.hpp
			src/math/vector3.hpp

			# tsdf: utilities for generating TSDF voxel grids in various ways
			src/tsdf/ewa_2d.cpp
			src/tsdf/ewa_3d.cpp
			src/tsdf/ewa_3d_viz.cpp
			src/tsdf/ewa_common.hpp
			src/tsdf/ewa.hpp

			# image_io: minimalistic image reading / writing
			src/image_io/stb_image.h
			src/image_io/stb_image_write.h
			src/image_io/imageio_stb_image.hpp
			src/image_io/imageio_stb_image.cpp
			src/image_io/png_eigen.hpp
			src/image_io/png_eigen.cpp

			# console: utilities for printing things to the console in better ways
			src/console/colors.hpp
			src/console/colors.cpp
			src/console/pretty_printers.hpp
			src/console/progress_bar.cpp
			src/console/progress_bar.hpp
			
			src/error_handling/throw_assert.hpp
			)

target_link_libraries(${MODULE_NAME}
        PUBLIC
        ${Boost_LIBRARIES}
        ${PYTHON3_LIBRARIES}
        ${GCCLIBATOMIC_LIBRARY}
)

target_link_libraries(${MODULE_NAME}
        PRIVATE
        OpenMP::OpenMP_CXX
)

target_include_directories(${MODULE_NAME} PUBLIC
		 ${PYTHON3_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${NUMPY_INCLUDE_DIRS}
)

if(NOT HUNTER_ENABLED)
	if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
	    # Provisions for typical Boost compiled on Windows
	    # Unless some extra compile options are used on Windows, the libraries won't have prefixes (change as necesssary)
	    target_compile_definitions(${MODULE_NAME} PUBLIC -DBOOST_ALL_NO_LIB -DBOOST_SYSTEM_NO_DEPRECATED)
	else()
	    target_compile_definitions(${MODULE_NAME} PUBLIC -DBOOST_TEST_DYN_LINK)
	endif()
endif()

# left for possible future CMake debugging
# TODO: print when CMake is envoked with debugger flags (see how it's done in other CMake projects)
#message(STATUS "${MODULE_NAME}: [archive output name:]  ${ARCHIVE_OUTPUT_NAME} [archive suffix:] ${${MODULE_NAME}_PY_SUFFIX}")
#message(STATUS "[include dirs:] ${PYTHON3_INCLUDE_DIRS} [packages path:] ${PYTHON3_PACKAGES_PATH} [libraries:] ${PYTHON3_LIBRARIES}")

#======== INSTALL/UNINSTALL TARGET =====================================================================================
set_target_properties(${MODULE_NAME} PROPERTIES
        ARCHIVE_OUTPUT_NAME ${ARCHIVE_OUTPUT_NAME}  # prevent name conflict for python2/3 outputs
        PREFIX ""
        OUTPUT_NAME ${MODULE_NAME}
        SUFFIX ${${MODULE_NAME}_PY_SUFFIX})

if ((CMAKE_CXX_COMPILER_ID MATCHES MSVC) AND (NOT PYTHON_DEBUG_LIBRARIES))
    set(PYTHON_INSTALL_CONFIGURATIONS CONFIGURATIONS Release)
else ()
    set(PYTHON_INSTALL_CONFIGURATIONS "")
endif ()

if (WIN32)
    set(PYTHON_INSTALL_ARCHIVE "")
else ()
    set(PYTHON_INSTALL_ARCHIVE ARCHIVE DESTINATION ${PYTHON3_PACKAGES_PATH} COMPONENT python)
endif ()

# install target
install(TARGETS ${MODULE_NAME}
        ${PYTHON_INSTALL_CONFIGURATIONS}
        RUNTIME DESTINATION ${PYTHON3_PACKAGES_PATH} COMPONENT python
        LIBRARY DESTINATION ${PYTHON3_PACKAGES_PATH} COMPONENT python
        ${PYTHON_INSTALL_ARCHIVE}
        )
        
# uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
# ================== COPY LIBRARY TO PYTHON DIRECTORY ABOVE ============================================================
add_custom_command(TARGET ${MODULE_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${MODULE_NAME}> ${PROJECT_SOURCE_DIR}/../$<TARGET_FILE_NAME:${MODULE_NAME}>
)
# ================== UNIT TESTING ======================================================================================
enable_testing()

# TODO: this is beginnning to look like a lot of DRY violations -- good source material for a CMake macro or function
# that would set up a single test based on the sources  

set (COMMON_TEST_SOURCES
	tests/common.hpp
	tests/common.cpp
)

add_executable(test_tsdf
	tests/test_tsdf.cpp
	tests/data/test_data_tsdf.hpp
	${COMMON_TEST_SOURCES}
)
add_executable(test_math
	tests/test_math.cpp
	tests/data/test_data_math.hpp
	${COMMON_TEST_SOURCES}
)
add_executable(test_slavcheva_optimizer
	tests/test_slavcheva_optimizer.cpp
	tests/data/test_data_slavcheva_optimizer.hpp
	${COMMON_TEST_SOURCES}
)
add_executable(test_hierarchical_optimizer
	tests/test_hierarchical_optimizer.cpp
	tests/data/test_data_hierarchical_optimizer.hpp
	${COMMON_TEST_SOURCES}
)
add_executable(test_image_io
	tests/test_image_io.cpp
	tests/data/test_data_image_io.hpp
	${COMMON_TEST_SOURCES}
)

target_link_libraries(test_tsdf ${MODULE_NAME} ${Boost_LIBRARIES})
target_include_directories(test_tsdf PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(test_math ${MODULE_NAME} ${Boost_LIBRARIES})
target_include_directories(test_math PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(test_slavcheva_optimizer ${MODULE_NAME} ${Boost_LIBRARIES})
target_include_directories(test_slavcheva_optimizer PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(test_hierarchical_optimizer ${MODULE_NAME} ${Boost_LIBRARIES})
target_include_directories(test_hierarchical_optimizer PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(test_image_io ${MODULE_NAME} ${Boost_LIBRARIES})
target_include_directories(test_image_io PRIVATE ${EIGEN3_INCLUDE_DIRS})

add_test(NAME image_io COMMAND test_image_io)
add_test(NAME tsdf COMMAND test_tsdf)
add_test(NAME math COMMAND test_math)
add_test(NAME slavcheva_optimizer COMMAND test_slavcheva_optimizer)
add_test(NAME hierarchical_optimizer COMMAND test_hierarchical_optimizer)

# copy test data
file(GLOB PNG_DATA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/*.png")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_data")
foreach(PNG_DATA_FILE ${PNG_DATA_FILES})
	get_filename_component(FILENAME ${PNG_DATA_FILE} NAME)
	configure_file(
		${PNG_DATA_FILE}
		"${CMAKE_CURRENT_BINARY_DIR}/test_data/${FILENAME}"
		COPYONLY
	)
endforeach(PNG_DATA_FILE)
