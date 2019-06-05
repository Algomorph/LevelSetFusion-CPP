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

#=================== TARGET DEFININTIONS ===============================================================================

# main library
set(MODULE_SOURCES
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
	src/python_export/telemetry.tpp
	src/python_export/telemetry.cpp
	src/python_export/math.hpp
	src/python_export/math.cpp
	src/python_export/numpy_conversions_shared.hpp
	src/python_export/numpy_conversions_shared.cpp
	src/python_export/slavcheva_optimizer.hpp
	src/python_export/slavcheva_optimizer.cpp
	src/python_export/tsdf.hpp
	src/python_export/tsdf.cpp
	src/python_export/sdf_2_sdf_optimizer.hpp
	src/python_export/sdf_2_sdf_optimizer.cpp

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
	src/telemetry/warp_delta_statistics.tpp
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
	src/math/cwise_binary.hpp
	src/math/cwise_binary.tpp
	src/math/cwise_binary.cpp
	src/math/extent.hpp
	src/math/field_like.hpp
	src/math/field_like.tpp
	src/math/field_like.cpp
	src/math/filtered_statistics.hpp
	src/math/filtered_statistics.tpp
	src/math/filtered_statistics.cpp
	src/math/gradients.hpp
	src/math/gradients.tpp
	src/math/gradients.cpp
	src/math/math_utils.hpp
	src/math/matrix_base.hpp
	src/math/matrix2.hpp
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
	src/math/transformation.hpp
    src/math/transformation.tpp
	src/math/transformation.cpp

	# tsdf: utilities for generating TSDF voxel grids in various ways
	src/tsdf/common.hpp
	src/tsdf/ewa_2d_viz.cpp
	src/tsdf/ewa_3d_viz.cpp
	src/tsdf/ewa_common.hpp
	src/tsdf/ewa_viz.hpp
	src/tsdf/generator.hpp
	src/tsdf/generator_crtp.hpp
	src/tsdf/generator_crtp.tpp
	src/tsdf/generator_crtp.cpp
	src/tsdf/generator_tensor.tpp
	src/tsdf/generator_tensor.cpp
	src/tsdf/generator_matrix.tpp
	src/tsdf/generator_matrix.cpp
	src/tsdf/parameters.hpp
	src/tsdf/parameters.tpp
	src/tsdf/parameters.cpp

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
	
	#rigid_optimization: rigid tracker from SDF-to-SDF fusion
	src/rigid_optimization/sdf_2_sdf_optimizer.hpp
	src/rigid_optimization/sdf_2_sdf_optimizer.tpp
	src/rigid_optimization/sdf_2_sdf_optimizer.cpp
	src/rigid_optimization/sdf_gradient_wrt_transformation.hpp
	src/rigid_optimization/sdf_gradient_wrt_transformation_matrix.tpp
	src/rigid_optimization/sdf_gradient_wrt_transformation_matrix.cpp
	src/rigid_optimization/sdf_gradient_wrt_transformation_tensor.tpp
	src/rigid_optimization/sdf_gradient_wrt_transformation_tensor.cpp
	src/rigid_optimization/sdf_weight.hpp
	src/rigid_optimization/sdf_weight_matrix.tpp
	src/rigid_optimization/sdf_weight_matrix.cpp
	src/rigid_optimization/sdf_weight_tensor.tpp
	src/rigid_optimization/sdf_weight_tensor.cpp

)

if (${CMAKE_VERSION} VERSION_LESS 3.8 OR (MSVC_IDE AND ${CMAKE_VERSION} VERSION_LESS 3.9))
    if (WITH_CUDA)
        cuda_add_library(${MODULE_NAME} SHARED ${MODULE_SOURCES})
    else ()
        add_library(${MODULE_NAME} SHARED ${MODULE_SOURCES})
    endif ()
else ()
    if (WITH_CUDA)
        enable_language(CUDA)
    endif ()

    add_library(${MODULE_NAME} SHARED ${MODULE_SOURCES})
    
    if (WITH_CUDA)
    	target_include_directories(${targetname} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
endif ()


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