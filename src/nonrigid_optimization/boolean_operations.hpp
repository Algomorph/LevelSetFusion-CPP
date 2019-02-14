//  ================================================================
//  Created by Gregory Kramida on 10/23/18.
//  Copyright (c) 2018 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================
#pragma once

#include <Eigen/Eigen>

namespace eig = Eigen;

namespace nonrigid_optimization{

	/**
	 * @param tsdf_value_a
	 * @param tsdf_value_b
	 * @param tolerance
	 * @return whether both SDF values is within tolerance of the truncation thresholds -1.0f and 1.0f
	 */
    inline bool are_both_SDF_values_truncated_tolerance(float tsdf_value_a, float tsdf_value_b, float tolerance = 10e-6f){
        return (1.0f - std::abs(tsdf_value_a) < tolerance && 1.0f - std::abs(tsdf_value_b) < tolerance);
    }
    /**
     * @param tsdf_value_a
     * @param tsdf_value_b
     * @return whether both SDF values have absolute value of 1.0f
     */
    inline bool are_both_SDF_values_truncated(float tsdf_value_a, float tsdf_value_b){
    	return std::abs(tsdf_value_a) == 1.0 && std::abs(tsdf_value_b) == 1.0;
    }
}// namespace boolean_ops
