#  ================================================================
#  Created by Gregory Kramida on 04/26/19.
#  Copyright (c) 2019 Gregory Kramida
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