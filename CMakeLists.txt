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

#=======================================================================================================================
#Hunter package discovery
hunter_add_package(Boost COMPONENTS test python)
hunter_add_package(Eigen)
#=======================================================================================================================
include(FindPackages)
#=================== DEBUG & VERBOSITY OPTIONS =========================================================================
option(SDF_GENERATION_CONSOLE_PROGRESS_REPORTS "SDF generation will print completion to stdout at fixed progress intervals." OFF)
option(ENABLE_CUDA "Build CUDA versions of things in addition to CPU versions if CUDA is available and discoverable." ON)
if(ENABLE_CUDA AND CUDA_FOUND)
	set(WITH_CUDA ON)
else()
	set(WITH_CUDA OFF)
	if(ENABLE_CUDA AND NOT CUDA_FOUND)
		message(WARNING "CUDA not found, disabling CUDA build.")
	endif()
endif()
#=================== HEADER CONFIGURATION ==============================================================================
configure_file(include/lsf_config.h.in include/lsf_config.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
#=======================================================================================================================
include(Targets)
include(Install_Uninstall)
include(Tests)
