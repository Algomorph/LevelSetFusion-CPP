/*
 * padding.tpp
 *
 *  Created on: Apr 8, 2019
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

#pragma once

//local
#include "padding.hpp"
#include "../error_handling/throw_assert.hpp"

namespace eig = Eigen;

namespace math {

template<typename Scalar>
Eigen::Tensor<Scalar, 3, Eigen::ColMajor> pad_replicate(const Eigen::Tensor<Scalar, 3, Eigen::ColMajor>& tensor,
		int border_width) {
	throw_assert(border_width > 0, "border_width should be a positive int value (>0). Got: " << border_width);

	typedef Eigen::Tensor<Scalar, 3, Eigen::ColMajor> TensorType;
	typedef eig::array<long, 3> Arrl3;

	int x_dim = tensor.dimension(0), y_dim = tensor.dimension(1), z_dim = tensor.dimension(2);

	int expand_by = border_width * 2;

	TensorType padded(x_dim + expand_by, y_dim + expand_by, z_dim + expand_by);

	int x_dim_padded = padded.dimension(0), y_dim_padded = padded.dimension(1), z_dim_padded = padded.dimension(2);

	Arrl3 offset = { border_width, border_width, border_width };
	Arrl3 extent = { x_dim, y_dim, z_dim };

	//_DEBUG
	padded.setZero();
	padded.slice(offset, extent) = tensor;

	// *** CORNERS ***
	extent = {border_width,border_width,border_width};
	TensorType corner(extent);
	//corner x=0, y=0, z=0
	corner.setConstant(tensor(0, 0, 0));
	offset = {0,0,0};
	padded.slice(offset, extent) = corner;
	//corner x=last,y=0,z=0
	corner.setConstant(tensor(x_dim - 1, 0, 0));
	offset = {x_dim+border_width, 0, 0};
	padded.slice(offset, extent) = corner;
	//corner x=0,y=last,z=0
	corner.setConstant(tensor(0, y_dim - 1, 0));
	offset = {0,y_dim+border_width,0};
	padded.slice(offset, extent) = corner;
	//corner x=last,y=last,z=0
	corner.setConstant(tensor(x_dim - 1, y_dim - 1, 0));
	offset = {x_dim+border_width,y_dim+border_width,0};
	padded.slice(offset, extent) = corner;
	//corner x=0, y=0, z=last
	corner.setConstant(tensor(0, 0, z_dim - 1));
	offset = {0,0,z_dim+border_width};
	padded.slice(offset, extent) = corner;
	//corner x=last,y=0,z=last
	corner.setConstant(tensor(x_dim - 1, 0, z_dim - 1));
	offset = {x_dim+border_width, 0, z_dim+border_width};
	padded.slice(offset, extent) = corner;
	//corner x=0,y=last,z=last
	corner.setConstant(tensor(0, y_dim - 1, z_dim - 1));
	offset = {0,y_dim+border_width,z_dim+border_width};
	padded.slice(offset, extent) = corner;
	//corner x=last,y=last,z=last
	corner.setConstant(tensor(x_dim - 1, y_dim - 1, z_dim - 1));
	offset = {x_dim+border_width,y_dim+border_width,z_dim+border_width};
	padded.slice(offset, extent) = corner;

	// *** FACES / SIDES ***
	// == xy plane ==
	extent= {x_dim,y_dim,1};
	TensorType face_xy(extent);
	//xy face_xy near, z = 0
	offset= {0,0,0};
	face_xy = tensor.slice(offset, extent);
	for (int i_border = 0; i_border < border_width; i_border++) {
		//for every face_xy, we make an offset to go beyond the border_width^3 corner
		offset= {border_width,border_width,i_border};
		padded.slice(offset,extent) = face_xy;
	}
	//xy face_xy far, z = last
	offset = {0,0,z_dim-1};
	face_xy = tensor.slice(offset, extent);
	for (int i_border = 0; i_border < border_width; i_border++) {
		offset= {border_width,border_width,z_dim_padded-i_border-1};
		padded.slice(offset,extent) = face_xy;
	}
	// == xz plane ==
	extent = {x_dim,1,z_dim};
	TensorType face_xz(extent);
	//xz face_xz at y = 0
	offset = {0,0,0};
	face_xz = tensor.slice(offset, extent);
	for (int i_border = 0; i_border < border_width; i_border++) {
		offset = {border_width,i_border,border_width};
		padded.slice(offset,extent) = face_xz;
	}
	//xz face_xz at y = last
	offset = {0,y_dim-1,0};
	face_xz = tensor.slice(offset, extent);
	for (int i_border = 0; i_border < border_width; i_border++) {
		offset = {border_width,y_dim_padded-i_border-1,border_width};
		padded.slice(offset,extent) = face_xz;
	}
	// == yz plane ==
	extent = {1,y_dim,z_dim};
	TensorType face_yz(extent);
	// yz face_yz at x = 0
	offset = {0,0,0};
	face_yz = tensor.slice(offset, extent);
	for (int i_border = 0; i_border < border_width; i_border++) {
		offset = {i_border, border_width, border_width};
		padded.slice(offset,extent) = face_yz;
	}
	// yz face_yz at x = last
	offset = {x_dim - 1, 0, 0};
	face_yz = tensor.slice(offset,extent);
	for (int i_border = 0; i_border < border_width; i_border++){
		offset = {x_dim_padded - i_border - 1, border_width, border_width};
		padded.slice(offset,extent) = face_yz;
	}

	// **** EDGES ****
	// === edges along the x axis ===
	extent = {x_dim, 1, 1};
	TensorType edge_along_x(extent);
	//edge at z = 0, y = 0
	offset = {0,0,0};
	edge_along_x = tensor.slice(offset, extent);
	for (int z_border = 0; z_border < border_width; z_border++) {
		for (int y_border = 0; y_border < border_width; y_border++) {
			offset = {border_width, y_border, z_border};
			padded.slice(offset,extent) = edge_along_x;
		}
	}
	//edge at z = 0, y = last
	offset = {0,y_dim-1,0};
	edge_along_x = tensor.slice(offset, extent);
	for (int z_border = 0; z_border < border_width; z_border++) {
		for (int y_border = 0; y_border < border_width; y_border++) {
			offset = {border_width, y_dim_padded - 1 - y_border, z_border};
			padded.slice(offset,extent) = edge_along_x;
		}
	}
	//edge at z = last, y = 0
	offset = {0,0,z_dim-1};
	edge_along_x = tensor.slice(offset, extent);
	for (int z_border = 0; z_border < border_width; z_border++) {
		for (int y_border = 0; y_border < border_width; y_border++) {
			offset = {border_width, y_border, z_dim_padded - 1 - z_border};
			padded.slice(offset,extent) = edge_along_x;
		}
	}
	//edge at z = last, y = last
	offset = {0,y_dim-1,z_dim-1};
	edge_along_x = tensor.slice(offset, extent);
	for (int z_border = 0; z_border < border_width; z_border++) {
		for (int y_border = 0; y_border < border_width; y_border++) {
			offset = {border_width, y_dim_padded - 1 - y_border, z_dim_padded - 1 - z_border};
			padded.slice(offset,extent) = edge_along_x;
		}
	}
	//=== edges along the y axis
	extent={1,y_dim,1};
	TensorType edge_along_y(extent);
	//edge at x = 0, z = 0
	offset = {0,0,0};
	edge_along_y = tensor.slice(offset, extent);
	for (int z_border = 0; z_border < border_width; z_border++) {
		for (int x_border = 0; x_border < border_width; x_border++) {
			offset = {x_border, border_width, z_border};
			padded.slice(offset,extent) = edge_along_y;
		}
	}
	//edge at x = last, z = 0
	offset = {x_dim-1,0,0};
	edge_along_y = tensor.slice(offset,extent);
	for (int z_border = 0; z_border < border_width; z_border++) {
		for (int x_border = 0; x_border < border_width; x_border++) {
			offset = {x_dim_padded - 1 - x_border, border_width, z_border};
			padded.slice(offset,extent) = edge_along_y;
		}
	}
	//edge at x = 0, z = last
	offset = {0,0,z_dim-1};
	edge_along_y = tensor.slice(offset, extent);
	for (int z_border = 0; z_border < border_width; z_border++) {
		for (int x_border = 0; x_border < border_width; x_border++) {
			offset = {x_border, border_width, z_dim_padded - 1 - z_border};
			padded.slice(offset,extent) = edge_along_y;
		}
	}
	//edge at x = last, z = last
	offset = {x_dim-1,0,z_dim-1};
	edge_along_y = tensor.slice(offset,extent);
	for (int z_border = 0; z_border < border_width; z_border++) {
		for (int x_border = 0; x_border < border_width; x_border++) {
			offset = {x_dim_padded - 1 - x_border, border_width, z_dim_padded - 1 - z_border};
			padded.slice(offset,extent) = edge_along_y;
		}
	}
	//=== edges along the z axis
	extent = {1,1,z_dim};
	TensorType edge_along_z(extent);
	//edge at x = 0, y = 0
	offset = {0,0,0};
	edge_along_z = tensor.slice(offset,extent);
	for (int y_border = 0; y_border < border_width; y_border++){
		for (int x_border = 0; x_border < border_width; x_border++){
			offset = {x_border, y_border, border_width};
			padded.slice(offset,extent) = edge_along_z;
		}
	}
	//edge at x = last, y = 0
	offset = {x_dim-1,0,0};
	edge_along_z = tensor.slice(offset,extent);
	for (int y_border = 0; y_border < border_width; y_border++){
		for (int x_border = 0; x_border < border_width; x_border++){
			offset = {x_dim_padded - 1 - x_border, y_border, border_width};
			padded.slice(offset,extent) = edge_along_z;
		}
	}
	//edge at x = 0, y = last
	offset = {0,y_dim-1,0};
	edge_along_z = tensor.slice(offset,extent);
	for (int y_border = 0; y_border < border_width; y_border++){
		for (int x_border = 0; x_border < border_width; x_border++){
			offset = {x_border, y_dim_padded - 1 - y_border, border_width};
			padded.slice(offset,extent) = edge_along_z;
		}
	}
	//edge at x = last, y = last
	offset = {x_dim-1,y_dim-1,0};
	edge_along_z = tensor.slice(offset,extent);
	for (int y_border = 0; y_border < border_width; y_border++){
		for (int x_border = 0; x_border < border_width; x_border++){
			offset = {x_dim_padded - 1 - x_border, y_dim_padded - 1 - y_border, border_width};
			padded.slice(offset,extent) = edge_along_z;
		}
	}


	return padded;
}

}
//end namespace math
