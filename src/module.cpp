//stdlib
#include <iostream>

//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

//local
#include <eigen_numpy.hpp>
#include "python_export/math.hpp"
#include "python_export/slavcheva_optimizer.hpp"
#include "python_export/hierarchical_optimizer.hpp"
#include "python_export/tsdf.hpp"

namespace bp = boost::python;
namespace pe = python_export;

Eigen::MatrixXd matrix_product_double(Eigen::MatrixXd a, Eigen::MatrixXd b) {
	return a * b;
}

Eigen::MatrixXf matrix_product_float(Eigen::MatrixXf a, Eigen::MatrixXf b) {
	return a * b;
}



BOOST_PYTHON_MODULE (level_set_fusion_optimization)
{
	SetupEigenConverters();

	// test functions
	bp::def("matrix_product_float64", matrix_product_double);
	bp::def("matrix_product_float32", matrix_product_float);

	pe::export_math_types();
	pe::export_math_functions();

	pe::export_ewa();

	pe::slavcheva::export_auxiliary_functions();
	pe::slavcheva::export_setting_singletons();
	pe::slavcheva::export_logging_utilities();
	pe::slavcheva::export_algorithms();

	pe::hierarchical_optimizer::export_algorithms();



	// endregion
}
