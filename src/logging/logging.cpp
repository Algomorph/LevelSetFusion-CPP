/*
 * logging.cpp
 *
 *  Created on: Nov 13, 2018
 *      Author: Gregory Kramida
 *   Copyright: 2018 Gregory Kramida
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
#include "logging.hpp"
#include "../math/filtered_statistics.hpp"
#include "../math/statistics.hpp"
#include "../math/assessment.hpp"

namespace logging {
WarpDeltaStatistics::WarpDeltaStatistics(
		float ratio_above_min_threshold,
		float length_min,
		float length_max,
		float length_mean,
		float length_standard_deviation,
		math::Vector2i longest_warp_location,
		bool is_largest_below_min_threshold,
		bool is_largest_above_max_threshold
		)
:
		ratio_above_min_threshold(ratio_above_min_threshold),
				length_min(length_min),
				length_max(length_max),
				length_mean(length_mean),
				length_standard_deviation(length_standard_deviation),
				longest_warp_location(longest_warp_location),
				is_largest_below_min_threshold(is_largest_below_min_threshold),
				is_largest_above_max_threshold(is_largest_above_max_threshold)
{
}

WarpDeltaStatistics::WarpDeltaStatistics(
		const math::MatrixXv2f& warp_field,
		const eig::MatrixXf& canonical_field,
		const eig::MatrixXf& live_field,
		float min_threshold, float max_threshold) {
	float length_max;
	math::Vector2i longest_warp_location;
	math::locate_max_norm(length_max, longest_warp_location, warp_field);
	float length_mean, standard_deviation, ratio_above_min_threshold;
	math::mean_and_std_vector_length_band_union(length_mean, standard_deviation, warp_field, live_field,
			canonical_field);
	float length_min = math::min_norm(warp_field);
	ratio_above_min_threshold =
			math::ratio_of_vector_lengths_above_threshold_band_union(warp_field,
					min_threshold, live_field,
					canonical_field);

	this->ratio_above_min_threshold = ratio_above_min_threshold;
	this->length_min = length_min;
	this->length_max = length_max;
	this->length_mean = length_mean;
	this->length_standard_deviation = standard_deviation;
	this->longest_warp_location = longest_warp_location;
	this->is_largest_below_min_threshold = length_max < min_threshold;
	this->is_largest_above_max_threshold = length_max > max_threshold;
}

eig::VectorXf WarpDeltaStatistics::to_array() {
	eig::VectorXf out(7);
	out << ratio_above_min_threshold,
			length_min,
			length_max,
			length_mean,
			length_standard_deviation,
			static_cast<float>(longest_warp_location.x),
			static_cast<float>(longest_warp_location.y),
			static_cast<float>(is_largest_below_min_threshold),
			static_cast<float>(is_largest_above_max_threshold)
	;
	return out;
}

bool WarpDeltaStatistics::operator==(const WarpDeltaStatistics & rhs) {
	return math::almost_equal(this->ratio_above_min_threshold, rhs.ratio_above_min_threshold) &&
			math::almost_equal(this->length_min, rhs.length_min) &&
			math::almost_equal(this->length_max, rhs.length_max) &&
			math::almost_equal(this->length_mean, rhs.length_mean) &&
			math::almost_equal(this->length_standard_deviation, rhs.length_standard_deviation) &&
			this->longest_warp_location == rhs.longest_warp_location &&
			this->is_largest_below_min_threshold == rhs.is_largest_below_min_threshold &&
			this->is_largest_above_max_threshold == rhs.is_largest_above_max_threshold
	;
}

bool WarpDeltaStatistics::operator!=(const WarpDeltaStatistics & rhs) {
	return !(*this == rhs);
}
//TODO: print out one-line JSON representation instead
std::ostream& operator<<(std::ostream &ostr, const WarpDeltaStatistics &ts)
		{
	ostr << "[warp delta stats]"
			<< std::endl << "  ratio: " << ts.ratio_above_min_threshold
			<< std::endl << "  min: " << ts.length_min
			<< std::endl << "  max: " << ts.length_max
			<< std::endl << "  mean: " << ts.length_mean
			<< std::endl << "  std: " << ts.length_standard_deviation
			<< std::endl << "  longest at: (" << ts.longest_warp_location << ")"
			<< std::endl << "  below min thresh: " << ts.is_largest_below_min_threshold
			<< std::endl << "  above max thresh: " << ts.is_largest_above_max_threshold
			;
	return ostr;
}

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

ConvergenceReport::ConvergenceReport(int iteration_count,
		bool iteration_limit_reached,
		WarpDeltaStatistics warp_delta_statistics,
		TsdfDifferenceStatistics tsdf_difference_statistics)
:
		iteration_count(iteration_count),
				iteration_limit_reached(iteration_limit_reached),
				warp_delta_statistics(warp_delta_statistics),
				tsdf_difference_statistics(tsdf_difference_statistics) {
}

std::ostream &operator<<(std::ostream &ostr, const ConvergenceReport &ts){
	ostr << "===[convergence report]==="
			<< std::endl << "  iter count: " << ts.iteration_count
			<< std::endl << "  limit reached: " << ts.iteration_limit_reached
			<< std::endl << "--------------------------"
			<< std::endl << ts.warp_delta_statistics
			<< std::endl << "--------------------------"
			<< std::endl << ts.tsdf_difference_statistics
			<< std::endl << "==========================)"
			;
	return ostr;
}

} //namespace logging

