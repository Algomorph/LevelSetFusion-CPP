/*
 * sdf_2_sdf_optimizer.cpp
 *
 *  Created on: Mar 21, 2019
 *      Author: Fei Shan
 */
//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
//local
#include "../rigid_optimization/sdf_2_sdf_optimizer2d.hpp"
#include "eigen_numpy.hpp"

namespace bp = boost::python;
namespace ro = rigid_optimization;

namespace python_export {
    namespace sdf_2_sdf_optimizer {
        void export_algorithms() {
            {
                bp::scope outer =
                        bp::class_<ro::Sdf2SdfOptimizer2d>("Sdf2SdfOptimizer2d",
                                bp::init<bp::optional<
                                        float, int,
                                        ro::Sdf2SdfOptimizer2d::VerbosityParameters>>(
                                        bp::args("rate",
                                                 "maximum_iteration_count",
                                                 "verbosity_parameters")))
                                .def("optimize", &ro::Sdf2SdfOptimizer2d::optimize,
                                     "Find optimal twist to map given live SDF to given canonical SDF",
                                     bp::args("image_y_coordinate",
                                              "canonical_depth_image",
                                              "live_depth_image",
                                              "tsdf_generation_parameters",
                                              "eta"));
                bp::class_<ro::Sdf2SdfOptimizer2d::VerbosityParameters>("VerbosityParameters",
                        "Parameters that control verbosity to stdout. "
                        "Assumes being used in an \"immutable\" manner, i.e. just a structure that holds values",
                        bp::init<bp::optional<bool, bool>>(
                                bp::args(/*"self",*/
                                "print_iteration_max_warp_update",
                                "print_iteration_energy")))
                        .add_property("print_iteration_max_warp_update",
                                      &ro::Sdf2SdfOptimizer2d::VerbosityParameters
                                      ::print_iteration_max_warp_update)
                        .add_property("print_iteration_energy",
                                      &ro::Sdf2SdfOptimizer2d::VerbosityParameters
                                      ::print_iteration_energy)
                                //============================================
                        .add_property("print_per_iteration_info",
                                      &ro::Sdf2SdfOptimizer2d::VerbosityParameters
                                      ::print_per_iteration_info);
                bp::class_<ro::Sdf2SdfOptimizer2d::TSDFGenerationParameters>("TSDFGenerationParameters",
                        "Parameters that control truncated SDF field generation."
                        "Assumes being used in an \"immutable\" manner, i.e. just a structure that holds values",
                        bp::init<bp::optional<float, eig::Matrix3f, eig::Matrix4f, eig::Vector3i, int, float, int>>(
                                bp::args(/*"self",*/
                                        "depth_unit_ratio",
                                        "camera_intrinsic_matrix",
                                        "camera_pose",
                                        "array_offset",
                                        "field_size",
                                        "voxel_size",
                                        "narrow_band_width_voxels")))
                        .add_property("depth_unit_ratio",
                                      &ro::Sdf2SdfOptimizer2d::TSDFGenerationParameters
                                      ::depth_unit_ratio)
                        .add_property("camera_intrinsic_matrix",
                              &ro::Sdf2SdfOptimizer2d::TSDFGenerationParameters
                              ::camera_intrinsic_matrix)
                        .add_property("camera_pose",
                              &ro::Sdf2SdfOptimizer2d::TSDFGenerationParameters
                              ::camera_pose)
                        .add_property("array_offset",
                                      &ro::Sdf2SdfOptimizer2d::TSDFGenerationParameters
                                      ::array_offset)
                        .add_property("field_size",
                                      &ro::Sdf2SdfOptimizer2d::TSDFGenerationParameters
                                      ::field_size)
                        .add_property("voxel_size",
                                      &ro::Sdf2SdfOptimizer2d::TSDFGenerationParameters
                                      ::voxel_size)
                        .add_property("narrow_band_width_voxels",
                                      &ro::Sdf2SdfOptimizer2d::TSDFGenerationParameters
                                      ::narrow_band_width_voxels);

            }
        }
    }
}