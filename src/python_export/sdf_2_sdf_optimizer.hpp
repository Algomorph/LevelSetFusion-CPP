/*
 * sdf_2_sdf_optimizer.hpp
 *
 *  Created on: Mar 21, 2019
 *      Author: Fei Shan
 */

#pragma once


namespace python_export{
namespace sdf_2_sdf_optimizer{

template<typename Scalar, typename ScalarContainer, typename TsdfGenerationParameters, typename TsdfGenerator, typename Transformation>
void export_algorithms(const char* suffix);

} // namespace hierarchical_optimizer
} // namespace python_export
