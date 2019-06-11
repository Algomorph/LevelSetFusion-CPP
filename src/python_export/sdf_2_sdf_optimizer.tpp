/*
 * sdf_2_sdf_optimizer.tpp
 *
 *  Created on: Mar 21, 2019
 *      Author: Fei Shan
 */
//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
//local
#include "../rigid_optimization/sdf_2_sdf_optimizer.hpp"
#include "../tsdf/parameters.hpp"
#include "eigen_numpy.hpp"


namespace bp = boost::python;
namespace ro = rigid_optimization;

namespace python_export {
namespace sdf_2_sdf_optimizer {

template<typename ScalarContainer, typename VectorContainer>
void export_algorithms(const char* suffix) {
    auto sufy = [&](const char* name) {
        return (std::string(name) + std::string(suffix));
    };
    bp::scope outer =
            bp::class_<ro::Sdf2SdfOptimizer<ScalarContainer, VectorContainer>>(sufy("Sdf2SdfOptimizer").c_str(),
                    bp::init<
                            bp::optional<
                                     float,
                                     int,
                                     typename tsdf::Parameters<ScalarContainer>,
                                     typename ro::Sdf2SdfOptimizer<ScalarContainer, VectorContainer>::VerbosityParameters>>(
                            bp::args("rate",
                                     "maximum_iteration_count",
                                     "tsdf_generation_parameters",
                                     "verbosity_parameters"
                                     )))
                    .def("optimize", &ro::Sdf2SdfOptimizer<ScalarContainer, VectorContainer>::optimize,
                    "Find optimal twist to map given live SDF to given canonical SDF",
                     bp::args("canonical_field",
                              "live_depth_image",
                              "eta",
                              "initial_camera_pose",
                              "image_y_coordinate"));
    bp::class_<typename ro::Sdf2SdfOptimizer<ScalarContainer, VectorContainer>::VerbosityParameters>(
            "VerbosityParameters",
            "Parameters that control verbosity to stdout. "
            "Assumes being used in an \"immutable\" manner, i.e. just a structure that holds values",
            bp::init<bp::optional<bool, bool>>(
                    bp::args(/*"self",*/
                            "print_iteration_max_warp_update",
                            "print_iteration_energy")))
            .add_property("print_iteration_max_warp_update",
                          &ro::Sdf2SdfOptimizer<ScalarContainer, VectorContainer>::VerbosityParameters
                          ::print_iteration_max_warp_update)
            .add_property("print_iteration_energy",
                          &ro::Sdf2SdfOptimizer<ScalarContainer, VectorContainer>::VerbosityParameters
                          ::print_iteration_energy)
                    //============================================
            .add_property("print_per_iteration_info",
                          &ro::Sdf2SdfOptimizer<ScalarContainer, VectorContainer>::VerbosityParameters
                          ::print_per_iteration_info);

}

}
}
