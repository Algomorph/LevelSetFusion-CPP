// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM
// Modified work copyright 2018-2019 Gregory Kramida
#pragma once

#include "math_utils.hpp"
#include "platform_independence.hpp"
#include "vector_base.hpp"

namespace math {
template<class T>
class Vector3 : public Vector3_<T> {
public:
	typedef T value_type;
	typedef T Scalar;
	static const int size = 3;
	_CPU_AND_GPU_CODE_ inline int get_size() const { return 3; }

	////////////////////////////////////////////////////////
	//  Constructors
	////////////////////////////////////////////////////////
	_CPU_AND_GPU_CODE_ Vector3() {} // Default constructor
	_CPU_AND_GPU_CODE_ explicit Vector3(const T& t) {
		this->x = t;
		this->y = t;
		this->z = t;
	}
	// Scalar constructor
	//TODO: ambiguity with above on calls like Vector2(0), provide meaningful alternative
//	_CPU_AND_GPU_CODE_ explicit Vector3(const T* tp) {
//		this->x = tp[0];
//		this->y = tp[1];
//		this->z = tp[2];
//	} // Construct from array
	_CPU_AND_GPU_CODE_ Vector3(const T v0, const T v1, const T v2) {
		this->x = v0;
		this->y = v1;
		this->z = v2;
	} // Construct from explicit values
	_CPU_AND_GPU_CODE_ explicit Vector3(const Vector4_<T>& u) {
		this->x = u.x;
		this->y = u.y;
		this->z = u.z;
	}

	_CPU_AND_GPU_CODE_ explicit Vector3(const Vector2_<T>& u, T v0 = T(0)) {
		this->x = u.x;
		this->y = u.y;
		this->z = v0;
	}

	_CPU_AND_GPU_CODE_ inline Vector3<int> to_int_round() const {
		return Vector3<int>((int) ROUND(this->x), (int) ROUND(this->y), (int) ROUND(this->z));
	}

	_CPU_AND_GPU_CODE_ inline Vector3<int> to_int() const {
		return Vector3<int>((int) (this->x), (int) (this->y), (int) (this->z));
	}

	_CPU_AND_GPU_CODE_ inline Vector3<int> to_int(Vector3<float>& residual) const {
		Vector3<int> intRound = to_int();
		residual = Vector3<float>(this->x - intRound.x, this->y - intRound.y, this->z - intRound.z);
		return intRound;
	}

	_CPU_AND_GPU_CODE_ inline Vector3<short> to_short_round() const {
		return Vector3<short>((short) ROUND(this->x), (short) ROUND(this->y), (short) ROUND(this->z));
	}

	_CPU_AND_GPU_CODE_ inline Vector3<short> to_short_floor() const {
		return Vector3<short>((short) floor(this->x), (short) floor(this->y), (short) floor(this->z));
	}

	_CPU_AND_GPU_CODE_ inline Vector3<int> to_int_floor() const {
		return Vector3<int>((int) floor(this->x), (int) floor(this->y), (int) floor(this->z));
	}

	_CPU_AND_GPU_CODE_ inline Vector3<int> to_int_floor(Vector3<float>& residual) const {
		Vector3<float> intFloor(floor(this->x), floor(this->y), floor(this->z));
		residual = *this - intFloor;
		return Vector3<int>((int) intFloor.x, (int) intFloor.y, (int) intFloor.z);
	}

	_CPU_AND_GPU_CODE_ inline Vector3<unsigned char> to_uchar() const {
		Vector3<int> vi = to_int_round();
		return Vector3<unsigned char>((unsigned char) CLAMP(vi.x, 0, 255), (unsigned char) CLAMP(vi.y, 0, 255),
		                              (unsigned char) CLAMP(vi.z, 0, 255));
	}

	_CPU_AND_GPU_CODE_ inline Vector3<float> to_float() const {
		return Vector3<float>((float) this->x, (float) this->y, (float) this->z);
	}

	_CPU_AND_GPU_CODE_ inline Vector3<double> to_double() const {
		return Vector3<double>((double) this->x, (double) this->y, (double) this->z);
	}

	_CPU_AND_GPU_CODE_ inline Vector3<float> normalised() const {
		float norm = 1.0f / sqrt((float) (this->x * this->x + this->y * this->y + this->z * this->z));
		return Vector3<float>((float) this->x * norm, (float) this->y * norm, (float) this->z * norm);
	}

	_CPU_AND_GPU_CODE_ const T* get_values() const { return this->values; }

	_CPU_AND_GPU_CODE_ Vector3<T>& set_values(const T* rhs) {
		this->x = rhs[0];
		this->y = rhs[1];
		this->z = rhs[2];
		return *this;
	}

	// indexing operators
	_CPU_AND_GPU_CODE_ T& operator[](int i) { return this->values[i]; }

	_CPU_AND_GPU_CODE_ const T& operator[](int i) const { return this->values[i]; }

	// type-cast operators
	_CPU_AND_GPU_CODE_ operator T*() { return this->values; }

	_CPU_AND_GPU_CODE_ operator const T*() const { return this->values; }

	////////////////////////////////////////////////////////
	//  Math operators
	////////////////////////////////////////////////////////

	// scalar multiply assign
	_CPU_AND_GPU_CODE_ friend Vector3<T>& operator*=(Vector3<T>& lhs, T d) {
		lhs.x *= d;
		lhs.y *= d;
		lhs.z *= d;
		return lhs;
	}

	// component-wise vector multiply assign
	_CPU_AND_GPU_CODE_ friend Vector3<T>& operator*=(Vector3<T>& lhs, const Vector3<T>& rhs) {
		lhs.x *= rhs.x;
		lhs.y *= rhs.y;
		lhs.z *= rhs.z;
		return lhs;
	}

	// scalar divide assign
	_CPU_AND_GPU_CODE_ friend Vector3<T>& operator/=(Vector3<T>& lhs, T d) {
		lhs.x /= d;
		lhs.y /= d;
		lhs.z /= d;
		return lhs;
	}

	// component-wise vector divide assign
	_CPU_AND_GPU_CODE_ friend Vector3<T>& operator/=(Vector3<T>& lhs, const Vector3<T>& rhs) {
		lhs.x /= rhs.x;
		lhs.y /= rhs.y;
		lhs.z /= rhs.z;
		return lhs;
	}

	// component-wise vector add assign
	_CPU_AND_GPU_CODE_ friend Vector3<T>& operator+=(Vector3<T>& lhs, const Vector3<T>& rhs) {
		lhs.x += rhs.x;
		lhs.y += rhs.y;
		lhs.z += rhs.z;
		return lhs;
	}

	// component-wise vector subtract assign
	_CPU_AND_GPU_CODE_ friend Vector3<T>& operator-=(Vector3<T>& lhs, const Vector3<T>& rhs) {
		lhs.x -= rhs.x;
		lhs.y -= rhs.y;
		lhs.z -= rhs.z;
		return lhs;
	}

	// unary negate
	_CPU_AND_GPU_CODE_ friend Vector3<T> operator-(const Vector3<T>& rhs) {
		Vector3<T> rv;
		rv.x = -rhs.x;
		rv.y = -rhs.y;
		rv.z = -rhs.z;
		return rv;
	}

	// vector add
	_CPU_AND_GPU_CODE_ friend Vector3<T> operator+(const Vector3<T>& lhs, const Vector3<T>& rhs) {
		Vector3<T> rv(lhs);
		return rv += rhs;
	}

	// vector subtract
	_CPU_AND_GPU_CODE_ friend Vector3<T> operator-(const Vector3<T>& lhs, const Vector3<T>& rhs) {
		Vector3<T> rv(lhs);
		return rv -= rhs;
	}

	// scalar multiply
	_CPU_AND_GPU_CODE_ friend Vector3<T> operator*(const Vector3<T>& lhs, T rhs) {
		Vector3<T> rv(lhs);
		return rv *= rhs;
	}

	// scalar multiply
	_CPU_AND_GPU_CODE_ friend Vector3<T> operator*(T lhs, const Vector3<T>& rhs) {
		Vector3<T> rv(lhs);
		return rv *= rhs;
	}

	// vector component-wise multiply
	_CPU_AND_GPU_CODE_ friend Vector3<T> operator*(const Vector3<T>& lhs, const Vector3<T>& rhs) {
		Vector3<T> rv(lhs);
		return rv *= rhs;
	}

	// scalar multiply
	_CPU_AND_GPU_CODE_ friend Vector3<T> operator/(const Vector3<T>& lhs, T rhs) {
		Vector3<T> rv(lhs);
		return rv /= rhs;
	}

	// vector component-wise multiply
	_CPU_AND_GPU_CODE_ friend Vector3<T> operator/(const Vector3<T>& lhs, const Vector3<T>& rhs) {
		Vector3<T> rv(lhs);
		return rv /= rhs;
	}

	////////////////////////////////////////////////////////
	//  Comparison operators
	////////////////////////////////////////////////////////

	// inequality
	_CPU_AND_GPU_CODE_ friend bool operator!=(const Vector3<T>& lhs, const Vector3<T>& rhs) {
		return (lhs.x != rhs.x) || (lhs.y != rhs.y) || (lhs.z != rhs.z);
	}

	////////////////////////////////////////////////////////////////////////////////
	// dimension specific operations
	////////////////////////////////////////////////////////////////////////////////

	friend std::ostream& operator<<(std::ostream& os, const Vector3<T>& dt) {
		os << dt.x << ", " << dt.y << ", " << dt.z;
		return os;
	}
};

////////////////////////////////////////////////////////
//  Non-member comparison operators
////////////////////////////////////////////////////////

// equality
template<typename T1, typename T2>
_CPU_AND_GPU_CODE_ inline bool operator==(const Vector3<T1>& lhs, const Vector3<T2>& rhs) {
	return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
}

////////////////////////////////////////////////////////
//  Non-member functions
////////////////////////////////////////////////////////

// cross product
template<typename T>
_CPU_AND_GPU_CODE_ Vector3<T> cross(const Vector3<T>& lhs, const Vector3<T>& rhs) {
	Vector3<T> r;
	r.x = lhs.y * rhs.z - lhs.z * rhs.y;
	r.y = lhs.z * rhs.x - lhs.x * rhs.z;
	r.z = lhs.x * rhs.y - lhs.y * rhs.x;
	return r;
}
}//namespace math
