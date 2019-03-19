/*
 * tsdf_difference_statistics.cpp
 *
 *  Created on: Mar 15, 2019
 *      Author: Gregory Kramida
 *   Copyright: 2019 Gregory Kramida
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

//local
#include "../telemetry/tsdf_difference_statistics.hpp"

#include "../math/filtered_statistics.hpp"
#include "../math/statistics.hpp"
#include "../math/assessment.hpp"


namespace logging {
TsdfDifferenceStatistics::TsdfDifferenceStatistics(
		float difference_min,
		float difference_max,
		float difference_mean,
		float difference_standard_deviation,
		math::Vector2i biggest_difference_location
		) :
		difference_min(difference_min),
				difference_max(difference_max),
				difference_mean(difference_mean),
				difference_standard_deviation(difference_standard_deviation),
				biggest_difference_location(biggest_difference_location)
{
}

TsdfDifferenceStatistics::TsdfDifferenceStatistics(const eig::MatrixXf& canonical_field,
		const eig::MatrixXf& live_field) {
	eig::ArrayXXf diff = (live_field.array() - canonical_field.array()).abs().eval();
	float diff_min = diff.minCoeff();
	eig::ArrayXXf::Index max_row, max_col;
	float diff_max = diff.maxCoeff(&max_row, &max_col);
	float diff_mean = diff.mean();
	float diff_std = std::sqrt((diff - diff_mean).square().sum() / (static_cast<float>(diff.size()) - 1.0f));
	math::Vector2i diff_max_loc(max_col, max_row);

	this->difference_min = diff_min;
	this->difference_max = diff_max;
	this->difference_mean = diff_mean;
	this->difference_standard_deviation = diff_std;
	this->biggest_difference_location = diff_max_loc;
}

eig::VectorXf TsdfDifferenceStatistics::to_array() {
	eig::VectorXf out(7);
	out << difference_min,
			difference_max,
			difference_mean,
			difference_standard_deviation,
			static_cast<float>(biggest_difference_location.x),
			static_cast<float>(biggest_difference_location.y)
	;
	return out;
}

bool TsdfDifferenceStatistics::operator==(const TsdfDifferenceStatistics& rhs) {
	return math::almost_equal(this->difference_min, rhs.difference_min) &&
			math::almost_equal(this->difference_max, rhs.difference_max) &&
			math::almost_equal(this->difference_mean, rhs.difference_mean) &&
			math::almost_equal(this->difference_standard_deviation, rhs.difference_standard_deviation) &&
			this->biggest_difference_location == rhs.biggest_difference_location;
}

bool TsdfDifferenceStatistics::operator!=(const TsdfDifferenceStatistics& rhs) {
	return !(*this == rhs);
}

//TODO: print out one-line JSON representation instead
std::ostream &operator<<(std::ostream &ostr, const TsdfDifferenceStatistics &ts){
	ostr << "[tsdf diff stats]"
			<< std::endl << "  min: " << ts.difference_min
			<< std::endl << "  max: " << ts.difference_max
			<< std::endl << "  mean: " << ts.difference_mean
			<< std::endl << "  std: " << ts.difference_standard_deviation
			<< std::endl << "  greatest diff at: (" << ts.biggest_difference_location << ")"
			;
	return ostr;
}

} //namespace logging
