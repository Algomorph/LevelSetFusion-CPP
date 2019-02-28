// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2016 Daniele Panozzo <daniele.panozzo@gmail.com>
// Copyright Modified Work (C) 2019 Gregory Kramida <algomorph@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "png_eigen.hpp"

#include "imageio_stb_image.hpp"

bool imageio::png::read_RGBA(
		const std::string png_file,
		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& R,
		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& G,
		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& B,
		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& A
		){
	int cols, rows, n;
	unsigned char *data = stbi_load(png_file.c_str(), &cols, &rows, &n, 4);
	if (data == nullptr) {
		return false;
	}

	R.resize(cols, rows);
	G.resize(cols, rows);
	B.resize(cols, rows);
	A.resize(cols, rows);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			R(j, rows - 1 - i) = data[4 * (j + cols * i) + 0];
			G(j, rows - 1 - i) = data[4 * (j + cols * i) + 1];
			B(j, rows - 1 - i) = data[4 * (j + cols * i) + 2];
			A(j, rows - 1 - i) = data[4 * (j + cols * i) + 3];
		}
	}

	imageio::stbi_image_free(data);

	return true;
}

bool imageio::png::read_GRAY16(const std::string png_file,
		Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic>& D) {
	int cols, rows, n;
	unsigned short *data = stbi_load_16(png_file.c_str(), &cols, &rows, &n, 1);
	if (data == nullptr) {
		return false;
	}

	D.resize(rows,cols);
	for (int i_row = 0; i_row < rows; ++i_row) {
		for (int i_col = 0; i_col < cols; ++i_col) {
			D(i_row, i_col) = data[cols * i_row + i_col];
		}
	}

	imageio::stbi_image_free(data);

	return true;
}
