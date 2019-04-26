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