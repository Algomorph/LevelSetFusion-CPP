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

# ================== UNIT TESTING ======================================================================================
enable_testing()

# TODO: this is beginnning to look like a lot of DRY violations -- good source material for a CMake macro or function
# that would set up a single test based on the sources

set (COMMON_TEST_SOURCES
	tests/common.hpp
	tests/common.cpp
)

macro (LSF_ADD_TEST)
	set(options)
	set(oneValueArgs NAME)
	set(multiValueArgs SOURCES)
	cmake_parse_arguments(LSF_ADD_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
	add_executable(test_${LSF_ADD_TEST_NAME}
		${LSF_ADD_TEST_SOURCES}
		${COMMON_TEST_SOURCES}
	)
	target_link_libraries(test_${LSF_ADD_TEST_NAME} ${MODULE_NAME} ${Boost_LIBRARIES})
	target_include_directories(test_${LSF_ADD_TEST_NAME} PRIVATE ${EIGEN3_INCLUDE_DIRS})
	add_test(NAME ${LSF_ADD_TEST_NAME} COMMAND test_${LSF_ADD_TEST_NAME})
endmacro()

lsf_add_test(NAME checks SOURCES tests/test_checks.cpp)
lsf_add_test(NAME tsdf SOURCES tests/test_tsdf.cpp tests/data/test_data_tsdf.hpp)
lsf_add_test(NAME math SOURCES tests/test_math.cpp tests/data/test_data_math.hpp)
lsf_add_test(NAME convolution SOURCES tests/test_convolution.cpp tests/data/test_data_convolution.hpp)
lsf_add_test(NAME resampling SOURCES tests/test_resampling.cpp)
lsf_add_test(NAME gradients SOURCES tests/test_gradients.cpp tests/data/test_data_gradients.hpp)
lsf_add_test(NAME slavcheva_optimizer SOURCES tests/test_slavcheva_optimizer.cpp tests/data/test_data_slavcheva_optimizer.hpp)
lsf_add_test(NAME hierarchical_optimizer SOURCES tests/test_hierarchical_optimizer.cpp tests/data/test_data_hierarchical_optimizer.hpp)
lsf_add_test(NAME image_io SOURCES tests/test_image_io.cpp tests/data/test_data_image_io.hpp)
lsf_add_test(NAME sdf_2_sdf_optimizer SOURCES tests/test_sdf_2_sdf_optimizer.cpp)

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