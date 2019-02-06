//stdlib
#include <iostream>

//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

//local
#include <lsf_config.h>
#include "python_export/eigen_numpy.hpp"
#include "python_export/math.hpp"
#include "python_export/slavcheva_optimizer.hpp"
#include "python_export/hierarchical_optimizer.hpp"
#include "python_export/tsdf.hpp"
#include "python_export/conversion_tests.hpp"

namespace bp = boost::python;
namespace pe = python_export;

BOOST_PYTHON_MODULE ( MODULE_NAME )
{
	setup_Eigen_matrix_converters();
	setup_Eigen_tensor_converters();

	pe::export_conversion_tests();

	pe::export_math_types();
	pe::export_math_functions();

	pe::export_ewa();

	pe::slavcheva::export_auxiliary_functions();
	pe::slavcheva::export_setting_singletons();
	pe::slavcheva::export_logging_utilities();
	pe::slavcheva::export_algorithms();

	pe::hierarchical_optimizer::export_algorithms();
}
