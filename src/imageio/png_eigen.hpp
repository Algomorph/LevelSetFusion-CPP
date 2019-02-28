// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2016 Daniele Panozzo <daniele.panozzo@gmail.com>
// Copyright Modified Work (C) 2019 Gregory Kramida <algomorph@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef IGL_PNG_READ_PNG_H
#define IGL_PNG_READ_PNG_H
#include <Eigen/Core>
#include <string>

namespace imageio {
namespace png {
// Read an image from a .png file into 4 memory buffers
//
// Input:
//  png_file  path to .png file
// Output:
//  R,G,B,A texture channels
// Returns true on success, false on failure
//
bool read_RGBA(const std::string png_file,
		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& R,
		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& G,
		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& B,
		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& A
		);

bool read_GRAY16(const std::string png_file, Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic>& D);

} //namespace png
} //namespace imageio

#endif
