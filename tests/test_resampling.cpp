//  Created by Gregory Kramida on 04/16/18.
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

#define BOOST_TEST_MODULE test_resampling

//libraries
#include <boost/test/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Eigen>

//test targets
#include "../src/math/resampling.hpp"
#include "../src/math/almost_equal.hpp"
#include "../src/math/typedefs.hpp"


BOOST_AUTO_TEST_CASE(test_upsampling_linear_matrix01) {
	eig::MatrixXf input(2, 3);
	input << 1.0f, 2.0f, 3.0f,
			4.0f, 5.0f, 6.0f;

	eig::MatrixXf expected_output(4, 6);
	expected_output <<
			1.00f, 1.25f, 1.75f, 2.25f, 2.75f, 3.00f,
			1.75f, 2.00f, 2.50f, 3.00f, 3.50f, 3.75f,
			3.25f, 3.50f, 4.00f, 4.50f, 5.00f, 5.25f,
			4.00f, 4.25f, 4.75f, 5.25f, 5.75f, 6.00f;
	eig::MatrixXf output = math::upsampleX2_linear(input);
	BOOST_REQUIRE(math::almost_equal_verbose(output, expected_output, 1e-6));

	eig::MatrixXf input2(3, 2);
	input2 <<
			1.0f, 2.0f,
			3.0f, 4.0f,
			5.0f, 6.0f;

	eig::MatrixXf expected_output2(6, 4);
	expected_output2 <<
			1.00f, 1.25f, 1.75f, 2.00f,
			1.50f, 1.75f, 2.25f, 2.50f,
			2.50f, 2.75f, 3.25f, 3.50f,
			3.50f, 3.75f, 4.25f, 4.50f,
			4.50f, 4.75f, 5.25f, 5.50f,
			5.00f, 5.25f, 5.75f, 6.00f;

	eig::MatrixXf output2 = math::upsampleX2_linear(input2);
	BOOST_REQUIRE(math::almost_equal_verbose(output2, expected_output2, 1e-6));

	eig::MatrixXf input3(3, 3);
	input3 <<
			1.0f, 2.0f, 3.0f,
			4.0f, 5.0f, 6.0f,
			7.0f, 8.0f, 9.0f;
	eig::MatrixXf expected_output3(6, 6);
	expected_output3 <<
			1.00f, 1.25f, 1.75f, 2.25f, 2.75f, 3.00f,
			1.75f, 2.00f, 2.50f, 3.00f, 3.50f, 3.75f,
			3.25f, 3.50f, 4.00f, 4.50f, 5.00f, 5.25f,
			4.75f, 5.00f, 5.50f, 6.00f, 6.50f, 6.75f,
			6.25f, 6.50f, 7.00f, 7.50f, 8.00f, 8.25f,
			7.00f, 7.25f, 7.75f, 8.25f, 8.75f, 9.00f;
	eig::MatrixXf output3 = math::upsampleX2_linear(input3);
	BOOST_REQUIRE(math::almost_equal_verbose(output3, expected_output3, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_upsampling_nearest_tensor01) {
	math::Tensor3f input(2, 2, 2);
	input.setValues( //@formatter:off
		{{{ 1.0f, 2.0f },
		  { 3.0f, 4.0f }},
		 {{ 5.0f, 6.0f },
		  { 7.0f, 8.0f }}}
	);   //@formatter:on

	math::Tensor3f expected_output(4, 4, 4);
	expected_output.setValues( //@formatter:off
		{{{1.0f, 1.0f, 2.0f, 2.0f},
		  {1.0f, 1.0f, 2.0f, 2.0f},
		  {3.0f, 3.0f, 4.0f, 4.0f},
		  {3.0f, 3.0f, 4.0f, 4.0f}},
		 {{1.0f, 1.0f, 2.0f, 2.0f},
		  {1.0f, 1.0f, 2.0f, 2.0f},
		  {3.0f, 3.0f, 4.0f, 4.0f},
		  {3.0f, 3.0f, 4.0f, 4.0f}},
		 {{5.0f, 5.0f, 6.0f, 6.0f},
		  {5.0f, 5.0f, 6.0f, 6.0f},
		  {7.0f, 7.0f, 8.0f, 8.0f},
		  {7.0f, 7.0f, 8.0f, 8.0f}},
		 {{5.0f, 5.0f, 6.0f, 6.0f},
		  {5.0f, 5.0f, 6.0f, 6.0f},
		  {7.0f, 7.0f, 8.0f, 8.0f},
		  {7.0f, 7.0f, 8.0f, 8.0f}}}
	);   //@formatter:on

	math::Tensor3f output = math::upsampleX2_nearest(input);
	BOOST_REQUIRE(math::almost_equal_verbose(output, expected_output, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_upsampling_linear_tensor01) {
	math::Tensor3f input(2, 2, 2);
	input.setValues( //@formatter:off
		{{{ 1.0f, 5.0f },
		  { 3.0f, 7.0f }},
		 {{ 2.0f, 6.0f },
		  { 4.0f, 8.0f }}}
	);   //@formatter:on

	math::Tensor3f expected_output(4, 4, 4);
	expected_output.setValues( //@formatter:off
		{{{1.00f, 2.00f, 4.00f, 5.00f},
		  {1.50f, 2.50f, 4.50f, 5.50f},
		  {2.50f, 3.50f, 5.50f, 6.50f},
		  {3.00f, 4.00f, 6.00f, 7.00f}},
		 {{1.25f, 2.25f, 4.25f, 5.25f},
		  {1.75f, 2.75f, 4.75f, 5.75f},
		  {2.75f, 3.75f, 5.75f, 6.75f},
		  {3.25f, 4.25f, 6.25f, 7.25f}},
		 {{1.75f, 2.75f, 4.75f, 5.75f},
		  {2.25f, 3.25f, 5.25f, 6.25f},
		  {3.25f, 4.25f, 6.25f, 7.25f},
		  {3.75f, 4.75f, 6.75f, 7.75f}},
		 {{2.00f, 3.00f, 5.00f, 6.00f},
		  {2.50f, 3.50f, 5.50f, 6.50f},
		  {3.50f, 4.50f, 6.50f, 7.50f},
		  {4.00f, 5.00f, 7.00f, 8.00f}}}
	);   //@formatter:on

	math::Tensor3f output = math::upsampleX2_linear(input);
	BOOST_REQUIRE(math::almost_equal_verbose(output, expected_output, 1e-6));
	math::Tensor3f input2(3, 3, 3);
	input2.setValues( //@formatter:off
			{{{1.0f, 10.0f, 19.0f},
			  {4.0f, 13.0f, 22.0f},
			  {7.0f, 16.0f, 25.0f}},
			 {{2.0f, 11.0f, 20.0f},
			  {5.0f, 14.0f, 23.0f},
			  {8.0f, 17.0f, 26.0f}},
			 {{3.0f, 12.0f, 21.0f},
			  {6.0f, 15.0f, 24.0f},
			  {9.0f, 18.0f, 27.0f}}}
	);  //@formatter:on

	math::Tensor3f expected_output2(6, 6, 6);
	expected_output2.setValues( //@formatter:off
			{{{1.00f, 3.25f, 7.75f, 12.25f, 16.75f, 19.00f},
			  {1.75f, 4.00f, 8.50f, 13.00f, 17.50f, 19.75f},
			  {3.25f, 5.50f, 10.00f, 14.50f, 19.00f, 21.25f},
			  {4.75f, 7.00f, 11.50f, 16.00f, 20.50f, 22.75f},
			  {6.25f, 8.50f, 13.00f, 17.50f, 22.00f, 24.25f},
			  {7.00f, 9.25f, 13.75f, 18.25f, 22.75f, 25.00f}},
			 {{1.25f, 3.50f, 8.00f, 12.50f, 17.00f, 19.25f},
			  {2.00f, 4.25f, 8.75f, 13.25f, 17.75f, 20.00f},
			  {3.50f, 5.75f, 10.25f, 14.75f, 19.25f, 21.50f},
			  {5.00f, 7.25f, 11.75f, 16.25f, 20.75f, 23.00f},
			  {6.50f, 8.75f, 13.25f, 17.75f, 22.25f, 24.50f},
			  {7.25f, 9.50f, 14.00f, 18.50f, 23.00f, 25.25f}},
			 {{1.75f, 4.00f, 8.50f, 13.00f, 17.50f, 19.75f},
			  {2.50f, 4.75f, 9.25f, 13.75f, 18.25f, 20.50f},
			  {4.00f, 6.25f, 10.75f, 15.25f, 19.75f, 22.00f},
			  {5.50f, 7.75f, 12.25f, 16.75f, 21.25f, 23.50f},
			  {7.00f, 9.25f, 13.75f, 18.25f, 22.75f, 25.00f},
			  {7.75f, 10.00f, 14.50f, 19.00f, 23.50f, 25.75f}},
			 {{2.25f, 4.50f, 9.00f, 13.50f, 18.00f, 20.25f},
			  {3.00f, 5.25f, 9.75f, 14.25f, 18.75f, 21.00f},
			  {4.50f, 6.75f, 11.25f, 15.75f, 20.25f, 22.50f},
			  {6.00f, 8.25f, 12.75f, 17.25f, 21.75f, 24.00f},
			  {7.50f, 9.75f, 14.25f, 18.75f, 23.25f, 25.50f},
			  {8.25f, 10.50f, 15.00f, 19.50f, 24.00f, 26.25f}},
			 {{2.75f, 5.00f, 9.50f, 14.00f, 18.50f, 20.75f},
			  {3.50f, 5.75f, 10.25f, 14.75f, 19.25f, 21.50f},
			  {5.00f, 7.25f, 11.75f, 16.25f, 20.75f, 23.00f},
			  {6.50f, 8.75f, 13.25f, 17.75f, 22.25f, 24.50f},
			  {8.00f, 10.25f, 14.75f, 19.25f, 23.75f, 26.00f},
			  {8.75f, 11.00f, 15.50f, 20.00f, 24.50f, 26.75f}},
			 {{3.00f, 5.25f, 9.75f, 14.25f, 18.75f, 21.00f},
			  {3.75f, 6.00f, 10.50f, 15.00f, 19.50f, 21.75f},
			  {5.25f, 7.50f, 12.00f, 16.50f, 21.00f, 23.25f},
			  {6.75f, 9.00f, 13.50f, 18.00f, 22.50f, 24.75f},
			  {8.25f, 10.50f, 15.00f, 19.50f, 24.00f, 26.25f},
			  {9.00f, 11.25f, 15.75f, 20.25f, 24.75f, 27.00f}}}
	);  //@formatter:on
	math::Tensor3f output2 = math::upsampleX2_linear(input2);
	BOOST_REQUIRE(math::almost_equal_verbose(output2, expected_output2, 1e-6));
}
BOOST_AUTO_TEST_CASE(test_upsampling_linear_tensor02){
	math::Tensor3f input(4,3,3);
	input.setValues(//@formatter:off
			{{{1.0f,1.0f,1.0f},
			  {1.0f,1.0f,1.0f},
			  {1.0f,1.0f,1.0f}},
			 {{2.0f,2.0f,2.0f},
			  {2.0f,2.0f,2.0f},
			  {2.0f,2.0f,2.0f}},
			 {{3.0f,3.0f,3.0f},
			  {3.0f,3.0f,3.0f},
			  {3.0f,3.0f,3.0f}},
			 {{4.0f,4.0f,4.0f},
			  {4.0f,4.0f,4.0f},
			  {4.0f,4.0f,4.0f}}
	});//@formatter:on

	math::Tensor3f expected_output(8,6,6);
	expected_output.setValues(//@formatter:off
			{{{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
			  {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
			  {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
			  {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
			  {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
			  {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f}},
			 {{1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f},
			  {1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f},
			  {1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f},
			  {1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f},
			  {1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f},
			  {1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f}},
			 {{1.75f, 1.75f, 1.75f, 1.75f, 1.75f, 1.75f},
			  {1.75f, 1.75f, 1.75f, 1.75f, 1.75f, 1.75f},
			  {1.75f, 1.75f, 1.75f, 1.75f, 1.75f, 1.75f},
			  {1.75f, 1.75f, 1.75f, 1.75f, 1.75f, 1.75f},
			  {1.75f, 1.75f, 1.75f, 1.75f, 1.75f, 1.75f},
			  {1.75f, 1.75f, 1.75f, 1.75f, 1.75f, 1.75f}},
			 {{2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f},
			  {2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f},
			  {2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f},
			  {2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f},
			  {2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f},
			  {2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f}},
			 {{2.75f, 2.75f, 2.75f, 2.75f, 2.75f, 2.75f},
			  {2.75f, 2.75f, 2.75f, 2.75f, 2.75f, 2.75f},
			  {2.75f, 2.75f, 2.75f, 2.75f, 2.75f, 2.75f},
			  {2.75f, 2.75f, 2.75f, 2.75f, 2.75f, 2.75f},
			  {2.75f, 2.75f, 2.75f, 2.75f, 2.75f, 2.75f},
			  {2.75f, 2.75f, 2.75f, 2.75f, 2.75f, 2.75f}},
			 {{3.25f, 3.25f, 3.25f, 3.25f, 3.25f, 3.25f},
			  {3.25f, 3.25f, 3.25f, 3.25f, 3.25f, 3.25f},
			  {3.25f, 3.25f, 3.25f, 3.25f, 3.25f, 3.25f},
			  {3.25f, 3.25f, 3.25f, 3.25f, 3.25f, 3.25f},
			  {3.25f, 3.25f, 3.25f, 3.25f, 3.25f, 3.25f},
			  {3.25f, 3.25f, 3.25f, 3.25f, 3.25f, 3.25f}},
			 {{3.75f, 3.75f, 3.75f, 3.75f, 3.75f, 3.75f},
			  {3.75f, 3.75f, 3.75f, 3.75f, 3.75f, 3.75f},
			  {3.75f, 3.75f, 3.75f, 3.75f, 3.75f, 3.75f},
			  {3.75f, 3.75f, 3.75f, 3.75f, 3.75f, 3.75f},
			  {3.75f, 3.75f, 3.75f, 3.75f, 3.75f, 3.75f},
			  {3.75f, 3.75f, 3.75f, 3.75f, 3.75f, 3.75f}},
			 {{4.00f, 4.00f, 4.00f, 4.00f, 4.00f, 4.00f},
			  {4.00f, 4.00f, 4.00f, 4.00f, 4.00f, 4.00f},
			  {4.00f, 4.00f, 4.00f, 4.00f, 4.00f, 4.00f},
			  {4.00f, 4.00f, 4.00f, 4.00f, 4.00f, 4.00f},
			  {4.00f, 4.00f, 4.00f, 4.00f, 4.00f, 4.00f},
			  {4.00f, 4.00f, 4.00f, 4.00f, 4.00f, 4.00f}}}
	);//@formatter:on

	math::Tensor3f output = math::upsampleX2_linear(input);
	BOOST_REQUIRE(math::almost_equal_verbose(output, expected_output, 1e-6));
}


BOOST_AUTO_TEST_CASE(test_downsampling_linear_matrix01) {
	eig::MatrixXf input(6, 4);
	input <<
			0.0f, 1.0f, 2.0f, 3.0f,
			4.0f, 5.0f, 6.0f, 7.0f,
			8.0f, 9.0f, 10.0f, 11.0f,
			12.0f, 13.0f, 14.0f, 15.0f,
			16.0f, 17.0f, 18.0f, 19.0f,
			20.0f, 21.0f, 22.0f, 23.0f;
	eig::MatrixXf expected_output(3, 2);
	expected_output <<
			3.125f, 4.875f,
			10.625f, 12.375f,
			18.125f, 19.875f;
	eig::MatrixXf output = math::downsampleX2_linear(input);
	BOOST_REQUIRE(math::almost_equal_verbose(output, expected_output, 1e-6));

	eig::MatrixXf input2(8, 8);
	input2 <<
			0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
			8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
			16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
			24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
			32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
			40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,
			48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
			56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f;
	eig::MatrixXf expected_output2(4, 4);
	// NB: output is different from OpenCV at the border because OpenCV resize w/ INTER_LINEAR simply averages
	// each coarse border value from the 4 corresponding fine values, whereas this code treats the border values
	// as repeating and applies the same set of coefficients at the borders as any other place.
	expected_output2 <<
			5.625f, 7.5f, 9.5f, 11.375f,
			20.625f, 22.5f, 24.5f, 26.375f,
			36.625f, 38.5f, 40.5f, 42.375f,
			51.625f, 53.5f, 55.5f, 57.375f;

	eig::MatrixXf output2 = math::downsampleX2_linear(input2);
	BOOST_REQUIRE(math::almost_equal_verbose(output2, expected_output2, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_downsampling_linear_tensor01) {
	math::Tensor3f input(4, 4, 4);
	input.setValues(	//@formatter:off
		{{{1.0000, 2.0000, 4.0000, 5.0000},
		  {1.5000, 2.5000, 4.5000, 5.5000},
		  {2.5000, 3.5000, 5.5000, 6.5000},
		  {3.0000, 4.0000, 6.0000, 7.0000}},
		 {{1.2500, 2.2500, 4.2500, 5.2500},
		  {1.7500, 2.7500, 4.7500, 5.7500},
		  {2.7500, 3.7500, 5.7500, 6.7500},
		  {3.2500, 4.2500, 6.2500, 7.2500}},
		 {{1.7500, 2.7500, 4.7500, 5.7500},
		  {2.2500, 3.2500, 5.2500, 6.2500},
		  {3.2500, 4.2500, 6.2500, 7.2500},
		  {3.7500, 4.7500, 6.7500, 7.7500}},
		 {{2.0000, 3.0000, 5.0000, 6.0000},
		  {2.5000, 3.5000, 5.5000, 6.5000},
		  {3.5000, 4.5000, 6.5000, 7.5000},
		  {4.0000, 5.0000, 7.0000, 8.0000}}}
	); 		//@formatter:on
	math::Tensor3f expected_output(2,2,2);
	expected_output.setValues(//@formatter:off
			{{{2.3125f, 4.8125f},
			  {3.5625f, 6.0625f}},
			 {{2.9375f, 5.4375f},
			  {4.1875f, 6.6875f}}}
	);//@formatter:on
	math::Tensor3f output = math::downsampleX2_linear(input);
	BOOST_REQUIRE(math::almost_equal_verbose(output, expected_output, 1e-6));

	math::Tensor3f input2(6, 6, 6);
	input2.setValues( //@formatter:off
			{{{1.00f, 3.25f, 7.75f, 12.25f, 16.75f, 19.00f},
			  {1.75f, 4.00f, 8.50f, 13.00f, 17.50f, 19.75f},
			  {3.25f, 5.50f, 10.00f, 14.50f, 19.00f, 21.25f},
			  {4.75f, 7.00f, 11.50f, 16.00f, 20.50f, 22.75f},
			  {6.25f, 8.50f, 13.00f, 17.50f, 22.00f, 24.25f},
			  {7.00f, 9.25f, 13.75f, 18.25f, 22.75f, 25.00f}},
			 {{1.25f, 3.50f, 8.00f, 12.50f, 17.00f, 19.25f},
			  {2.00f, 4.25f, 8.75f, 13.25f, 17.75f, 20.00f},
			  {3.50f, 5.75f, 10.25f, 14.75f, 19.25f, 21.50f},
			  {5.00f, 7.25f, 11.75f, 16.25f, 20.75f, 23.00f},
			  {6.50f, 8.75f, 13.25f, 17.75f, 22.25f, 24.50f},
			  {7.25f, 9.50f, 14.00f, 18.50f, 23.00f, 25.25f}},
			 {{1.75f, 4.00f, 8.50f, 13.00f, 17.50f, 19.75f},
			  {2.50f, 4.75f, 9.25f, 13.75f, 18.25f, 20.50f},
			  {4.00f, 6.25f, 10.75f, 15.25f, 19.75f, 22.00f},
			  {5.50f, 7.75f, 12.25f, 16.75f, 21.25f, 23.50f},
			  {7.00f, 9.25f, 13.75f, 18.25f, 22.75f, 25.00f},
			  {7.75f, 10.00f, 14.50f, 19.00f, 23.50f, 25.75f}},
			 {{2.25f, 4.50f, 9.00f, 13.50f, 18.00f, 20.25f},
			  {3.00f, 5.25f, 9.75f, 14.25f, 18.75f, 21.00f},
			  {4.50f, 6.75f, 11.25f, 15.75f, 20.25f, 22.50f},
			  {6.00f, 8.25f, 12.75f, 17.25f, 21.75f, 24.00f},
			  {7.50f, 9.75f, 14.25f, 18.75f, 23.25f, 25.50f},
			  {8.25f, 10.50f, 15.00f, 19.50f, 24.00f, 26.25f}},
			 {{2.75f, 5.00f, 9.50f, 14.00f, 18.50f, 20.75f},
			  {3.50f, 5.75f, 10.25f, 14.75f, 19.25f, 21.50f},
			  {5.00f, 7.25f, 11.75f, 16.25f, 20.75f, 23.00f},
			  {6.50f, 8.75f, 13.25f, 17.75f, 22.25f, 24.50f},
			  {8.00f, 10.25f, 14.75f, 19.25f, 23.75f, 26.00f},
			  {8.75f, 11.00f, 15.50f, 20.00f, 24.50f, 26.75f}},
			 {{3.00f, 5.25f, 9.75f, 14.25f, 18.75f, 21.00f},
			  {3.75f, 6.00f, 10.50f, 15.00f, 19.50f, 21.75f},
			  {5.25f, 7.50f, 12.00f, 16.50f, 21.00f, 23.25f},
			  {6.75f, 9.00f, 13.50f, 18.00f, 22.50f, 24.75f},
			  {8.25f, 10.50f, 15.00f, 19.50f, 24.00f, 26.25f},
			  {9.00f, 11.25f, 15.75f, 20.25f, 24.75f, 27.00f}}}
	);  //@formatter:on
	math::Tensor3f expected_output2(3,3,3);
	expected_output2.setValues(//@formatter:off
		{{{3.4375f, 10.7500f, 18.0625f},
		  {5.8750f, 13.1875f, 20.5000f},
		  {8.3125f, 15.6250f, 22.9375f}},
		 {{4.2500f, 11.5625f, 18.8750f},
		  {6.6875f, 14.0000f, 21.3125f},
		  {9.1250f, 16.4375f, 23.7500f}},
		 {{5.0625f, 12.3750f, 19.6875f},
		  {7.5000f, 14.8125f, 22.1250f},
		  {9.9375f, 17.2500f, 24.5625f}}}
	);//@formatter:on
	math::Tensor3f output2 = math::downsampleX2_linear(input2);
	BOOST_REQUIRE(math::almost_equal_verbose(output2, expected_output2, 1e-6));

	math::Tensor3f input3_up(4,3,3);
	input3_up.setValues(//@formatter:off
			{{{1.0f,1.0f,1.0f},
			  {1.0f,1.0f,1.0f},
			  {1.0f,1.0f,1.0f}},
			 {{2.0f,2.0f,2.0f},
			  {2.0f,2.0f,2.0f},
			  {2.0f,2.0f,2.0f}},
			 {{3.0f,3.0f,3.0f},
			  {3.0f,3.0f,3.0f},
			  {3.0f,3.0f,3.0f}},
			 {{4.0f,4.0f,4.0f},
			  {4.0f,4.0f,4.0f},
			  {4.0f,4.0f,4.0f}}
	});//@formatter:on
	math::Tensor3f input3_down = math::upsampleX2_linear(input3_up);
	math::Tensor3f expected_output3(4,3,3);
	expected_output3.setValues(//@formatter:off
			{{{1.1875f, 1.1875f, 1.1875f},
			  {1.1875f, 1.1875f, 1.1875f},
			  {1.1875f, 1.1875f, 1.1875f}},
			 {{2.0000f, 2.0000f, 2.0000f},
			  {2.0000f, 2.0000f, 2.0000f},
			  {2.0000f, 2.0000f, 2.0000f}},
			 {{3.0000f, 3.0000f, 3.0000f},
			  {3.0000f, 3.0000f, 3.0000f},
			  {3.0000f, 3.0000f, 3.0000f}},
			 {{3.8125f, 3.8125f, 3.8125f},
			  {3.8125f, 3.8125f, 3.8125f},
			  {3.8125f, 3.8125f, 3.8125f}}}
	);//@formatter:on
	math::Tensor3f output3 = math::downsampleX2_linear(input3_down);
	BOOST_REQUIRE(math::almost_equal_verbose(output3, expected_output3, 1e-6));
}

