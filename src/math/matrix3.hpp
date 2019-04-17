// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM
// Modified work copyright 2019 Gregory Kramida
#pragma once

//stdlib
#include <numeric>

//local
#include "matrix_base.hpp"
#include "vector2.hpp"
#include "vector3.hpp"
#include "platform_independence.hpp"

namespace math {

//======================================================================================================================
//                        Matrix3 class with math operators (inspired by InfiniTAM ORUtils)
//======================================================================================================================

template<class T>
class Matrix3 : public Matrix3_< T >
{
public:
	_CPU_AND_GPU_CODE_ Matrix3() {}
	_CPU_AND_GPU_CODE_ Matrix3(T t) { setValues(t); }
	_CPU_AND_GPU_CODE_ Matrix3(const T *m)	{ setValues(m); }
	_CPU_AND_GPU_CODE_ Matrix3(T a00, T a01, T a02, T a10, T a11, T a12, T a20, T a21, T a22)	{
		this->m00 = a00; this->m01 = a01; this->m02 = a02;
		this->m10 = a10; this->m11 = a11; this->m12 = a12;
		this->m20 = a20; this->m21 = a21; this->m22 = a22;
	}

	_CPU_AND_GPU_CODE_ inline void get_values(T *mp) const	{ memcpy(mp, this->values, sizeof(T) * 9); }
	_CPU_AND_GPU_CODE_ inline const T *get_values() const { return this->values; }
	_CPU_AND_GPU_CODE_ inline Vector3<T> get_scale() const { return Vector3<T>(this->m00, this->m11, this->m22); }

	// Element access
	_CPU_AND_GPU_CODE_ inline T &operator()(int x, int y)	{ return at(x, y); }
	_CPU_AND_GPU_CODE_ inline const T &operator()(int x, int y) const	{ return at(x, y); }
	_CPU_AND_GPU_CODE_ inline T &operator()(Vector2<int> pnt)	{ return at(pnt.x, pnt.y); }
	_CPU_AND_GPU_CODE_ inline const T &operator()(Vector2<int> pnt) const	{ return at(pnt.x, pnt.y); }
	_CPU_AND_GPU_CODE_ inline T &at(int x, int y) { return this->m[x * 3 + y]; }
	_CPU_AND_GPU_CODE_ inline const T &at(int x, int y) const { return this->values[x * 3 + y]; }

	// set values
	_CPU_AND_GPU_CODE_ inline void set_values(const T *mp) { memcpy(this->values, mp, sizeof(T) * 9); }
	_CPU_AND_GPU_CODE_ inline void set_values(const T r)	{ for (int i = 0; i < 9; i++)	this->m[i] = r; }
	_CPU_AND_GPU_CODE_ inline void set_zeros() { memset(this->values, 0, sizeof(T) * 9); }
	_CPU_AND_GPU_CODE_ inline void set_identity() { set_zeros(); this->m00 = this->m11 = this->m22 = 1; }
	_CPU_AND_GPU_CODE_ inline void set_scale(T s) { this->m00 = this->m11 = this->m22 = s; }
	_CPU_AND_GPU_CODE_ inline void set_scale(const Vector3_<T> &s) { this->m00 = s[0]; this->m11 = s[1]; this->m22 = s[2]; }
	_CPU_AND_GPU_CODE_ inline void set_row(int r, const Vector3_<T> &t){ for (int x = 0; x < 3; x++) at(x, r) = t[x]; }
	_CPU_AND_GPU_CODE_ inline void set_column(int c, const Vector3_<T> &t) { memcpy(this->values + 3 * c, t.values, sizeof(T) * 3); }

	// get values
	_CPU_AND_GPU_CODE_ inline Vector3<T> get_row(int r) const { Vector3<T> v; for (int x = 0; x < 3; x++) v[x] = at(x, r); return v; }
	_CPU_AND_GPU_CODE_ inline Vector3<T> get_column(int c) const { Vector3<T> v; memcpy(v.values, this->m + 3 * c, sizeof(T) * 3); return v; }
	_CPU_AND_GPU_CODE_ inline Matrix3 t() { // transpose
		Matrix3 mtrans;
		for (int x = 0; x < 3; x++)	for (int y = 0; y < 3; y++)
			mtrans(x, y) = at(y, x);
		return mtrans;
	}

	_CPU_AND_GPU_CODE_ inline friend Matrix3 operator * (const Matrix3 &lhs, const Matrix3 &rhs)	{
		Matrix3 r;
		r.set_zeros();
		for (int x = 0; x < 3; x++) for (int y = 0; y < 3; y++) for (int k = 0; k < 3; k++)
			r(x, y) += lhs(k, y) * rhs(x, k);
		return r;
	}

	_CPU_AND_GPU_CODE_ inline friend Matrix3 operator + (const Matrix3 &lhs, const Matrix3 &rhs) {
		Matrix3 res(lhs.m);
		return res += rhs;
	}

	_CPU_AND_GPU_CODE_ inline friend Matrix3 operator - (const Matrix3 &lhs, const Matrix3 &rhs) {
		Matrix3 res(lhs.m);
		return res -= rhs;
	}

	_CPU_AND_GPU_CODE_ inline Vector3<T> operator *(const Vector3<T> &rhs) const {
		Vector3<T> r;
		r[0] = this->m[0] * rhs[0] + this->m[3] * rhs[1] + this->m[6] * rhs[2];
		r[1] = this->m[1] * rhs[0] + this->m[4] * rhs[1] + this->m[7] * rhs[2];
		r[2] = this->m[2] * rhs[0] + this->m[5] * rhs[1] + this->m[8] * rhs[2];
		return r;
	}

	_CPU_AND_GPU_CODE_ inline Matrix3 operator *(const T &r) const {
		Matrix3 res(this->m);
		return res *= r;
	}

	_CPU_AND_GPU_CODE_ inline friend Vector3<T> operator *(const Vector3<T> &lhs, const Matrix3 &rhs){
		Vector3<T> r;
		for (int x = 0; x < 3; x++)
			r[x] = lhs[0] * rhs(x, 0) + lhs[1] * rhs(x, 1) + lhs[2] * rhs(x, 2);
		return r;
	}

	_CPU_AND_GPU_CODE_ inline Matrix3& operator += (const T &r) { for (int i = 0; i < 9; ++i) this->m[i] += r; return *this; }
	_CPU_AND_GPU_CODE_ inline Matrix3& operator -= (const T &r) { for (int i = 0; i < 9; ++i) this->m[i] -= r; return *this; }
	_CPU_AND_GPU_CODE_ inline Matrix3& operator *= (const T &r) { for (int i = 0; i < 9; ++i) this->m[i] *= r; return *this; }
	_CPU_AND_GPU_CODE_ inline Matrix3& operator /= (const T &r) { for (int i = 0; i < 9; ++i) this->m[i] /= r; return *this; }
	_CPU_AND_GPU_CODE_ inline Matrix3& operator += (const Matrix3 &mat) { for (int i = 0; i < 9; ++i) this->m[i] += mat.m[i]; return *this; }
	_CPU_AND_GPU_CODE_ inline Matrix3& operator -= (const Matrix3 &mat) { for (int i = 0; i < 9; ++i) this->m[i] -= mat.m[i]; return *this; }

	_CPU_AND_GPU_CODE_ inline friend bool operator == (const Matrix3 &lhs, const Matrix3 &rhs) {
		bool r = lhs.m[0] == rhs.m[0];
		for (int i = 1; i < 9; i++)
			r &= lhs.m[i] == rhs.m[i];
		return r;
	}

	_CPU_AND_GPU_CODE_ inline friend bool operator != (const Matrix3 &lhs, const Matrix3 &rhs) {
		bool r = lhs.m[0] != rhs.m[0];
		for (int i = 1; i < 9; i++)
			r |= lhs.m[i] != rhs.m[i];
		return r;
	}

	// Sum of all elements
	_CPU_AND_GPU_CODE_ inline T sum() const { return std::accumulate(this->values, this->values+9, static_cast<T>(0)); }

	// Matrix determinant
	_CPU_AND_GPU_CODE_ inline T det() const {
		return (this->m11*this->m22 - this->m12*this->m21)*this->m00 + (this->m12*this->m20 - this->m10*this->m22)*this->m01 + (this->m10*this->m21 - this->m11*this->m20)*this->m02;
	}

	// The inverse matrix for float/double type
	_CPU_AND_GPU_CODE_ inline bool inv(Matrix3 &out) const {
		T determinant = det();
		if (determinant == 0) {
			out.set_zeros();
			return false;
		}

		out.m00 = (this->m11*this->m22 - this->m12*this->m21) / determinant;
		out.m01 = (this->m02*this->m21 - this->m01*this->m22) / determinant;
		out.m02 = (this->m01*this->m12 - this->m02*this->m11) / determinant;
		out.m10 = (this->m12*this->m20 - this->m10*this->m22) / determinant;
		out.m11 = (this->m00*this->m22 - this->m02*this->m20) / determinant;
		out.m12 = (this->m02*this->m10 - this->m00*this->m12) / determinant;
		out.m20 = (this->m10*this->m21 - this->m11*this->m20) / determinant;
		out.m21 = (this->m01*this->m20 - this->m00*this->m21) / determinant;
		out.m22 = (this->m00*this->m11 - this->m01*this->m10) / determinant;
		return true;
	}

	friend std::ostream& operator<<(std::ostream& os, const Matrix3<T>& dt)	{
		for (int y = 0; y < 3; y++)
			os << dt(0, y) << ", " << dt(1, y) << ", " << dt(2, y) << "\n";
		return os;
	}
};

}  // namespace math
