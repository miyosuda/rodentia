// -*- C++ -*-
#ifndef MATRIX4F_HEADER
#define MATRIX4F_HEADER

#include "Vector4f.h"
#include <math.h>

#include <stdio.h>

class Quat4f;

//====================================
//            [Matrix4f]
// 
// Member order is
//    [m00, m10, ....]
// Column major order.
// +------------------+
// |m00, m01, m02, m03|
// |m10, m11, m12, m13|
// |m20, m21, m22, m23|
// |m30, m31, m32, m33|
// +------------------+
//====================================
class Matrix4f {
public:
    float m00; // row1 colum1
    float m10; // row2 colum1
    float m20; // row3 colum1
    float m30; // row4 colum1
    
    float m01;
    float m11;
    float m21;
    float m31;
    
    float m02;
    float m12;
    float m22;
    float m32;
    
    float m03;
    float m13;
    float m23;
    float m33;
    
    Matrix4f() {}
    
    Matrix4f(const Matrix4f& m) {
        set(m);
    }

    Matrix4f(
        float m00_, float m10_, float m20_, float m30_,
        float m01_, float m11_, float m21_, float m31_,
        float m02_, float m12_, float m22_, float m32_,
        float m03_, float m13_, float m23_, float m33_
        ) {
        set(m00_, m10_, m20_, m30_,
            m01_, m11_, m21_, m31_,
            m02_, m12_, m22_, m32_,
            m03_, m13_, m23_, m33_);
    }   

    Matrix4f& operator=(const Matrix4f& m) {
        set(m);
        return *this; 
    }

    void set(
        float m00_, float m10_, float m20_, float m30_,
        float m01_, float m11_, float m21_, float m31_,
        float m02_, float m12_, float m22_, float m32_,
        float m03_, float m13_, float m23_, float m33_
        ) {
        m00=m00_; m10=m10_; m20=m20_; m30=m30_;
        m01=m01_; m11=m11_; m21=m21_; m31=m31_;
        m02=m02_; m12=m12_; m22=m22_; m32=m32_;
        m03=m03_; m13=m13_; m23=m23_; m33=m33_;
    }

    void set(const Matrix4f& m) {
        m00=m.m00; m10=m.m10; m20=m.m20; m30=m.m30;
        m01=m.m01; m11=m.m11; m21=m.m21; m31=m.m31;
        m02=m.m02; m12=m.m12; m22=m.m22; m32=m.m32;
        m03=m.m03; m13=m.m13; m23=m.m23; m33=m.m33;
    }

    void set( const Vector4f v0, 
              const Vector4f v1,
              const Vector4f v2,
              const Vector4f v3 ) {
        m00=v0.x; m10=v0.y; m20=v0.z; m30=v0.w;
        m01=v1.x; m11=v1.y; m21=v1.z; m31=v1.w;
        m02=v2.x; m12=v2.y; m22=v2.z; m32=v2.w;
        m03=v3.x; m13=v3.y; m23=v3.z; m33=v3.w;
    }

    // Implemented below
    void set(const Quat4f& q);

    void setElement(unsigned int row, unsigned int column, float f) {
        float* p = &m00;
        p[4*column + row] = f;
    }

    float getElement(unsigned int row, unsigned int column) const {
        const float* p = &m00;
        return p[4*column + row];
    }

    void setZero() {
        m00=m10=m20=m30 = 0.0f;
        m01=m11=m21=m31 = 0.0f;
        m02=m12=m22=m32 = 0.0f;
        m03=m13=m23=m33 = 0.0f;
    }

    void setIdentity() {
        m10=m20=m30 = 0.0f;
        m01=m21=m31 = 0.0f;
        m02=m12=m32 = 0.0f;
        m03=m13=m23 = 0.0f;     
        m00 = m11 = m22 = m33 = 1.0f;
    }

    void setRow(unsigned int row, const Vector4f& v) {
        float* p = &m00;
        p[row     ] = v.x;
        p[row + 4 ] = v.y;
        p[row + 8 ] = v.z;
        p[row + 12] = v.w;
    }

    void setColumn(unsigned int column, const Vector4f& v) {
        float* pbase = &m00;
        float* p = pbase + 4 * column;
        p[0] = v.x;
        p[1] = v.y;
        p[2] = v.z;
        p[3] = v.w;
    }

    void getRow(unsigned int row, Vector4f& v) const {
        const float* p = &m00;
        v.x = p[row     ];
        v.y = p[row + 4 ];
        v.z = p[row + 8 ];
        v.w = p[row + 12];
    }

    void getColumn(unsigned int column, Vector4f& v) const {
        const float* pbase = &m00;
        const float* p = pbase + 4 * column;
        v.x = p[0];
        v.y = p[1];
        v.z = p[2];
        v.w = p[3];
    }

    const Vector4f& getColumnRef(unsigned int column) const {
        return *( (const Vector4f*)(&m00 + 4*column) );
    }

    void add( const Matrix4f& m0, const Matrix4f& m1 ) {
        m00=m0.m00 + m1.m00; m10=m0.m10 + m1.m10; 
        m20=m0.m20 + m1.m20; m30=m0.m30 + m1.m30;
        m01=m0.m01 + m1.m01; m11=m0.m11 + m1.m11; 
        m21=m0.m21 + m1.m21; m31=m0.m31 + m1.m31;
        m02=m0.m02 + m1.m02; m12=m0.m12 + m1.m12; 
        m22=m0.m22 + m1.m22; m32=m0.m32 + m1.m32;
        m03=m0.m03 + m1.m03; m13=m0.m13 + m1.m13; 
        m23=m0.m23 + m1.m23; m33=m0.m33 + m1.m33;       
    }

    void sub( const Matrix4f& m0, const Matrix4f& m1 ) {
        m00=m0.m00 - m1.m00; m10=m0.m10 - m1.m10; 
        m20=m0.m20 - m1.m20; m30=m0.m30 - m1.m30;
        m01=m0.m01 - m1.m01; m11=m0.m11 - m1.m11; 
        m21=m0.m21 - m1.m21; m31=m0.m31 - m1.m31;
        m02=m0.m02 - m1.m02; m12=m0.m12 - m1.m12; 
        m22=m0.m22 - m1.m22; m32=m0.m32 - m1.m32;
        m03=m0.m03 - m1.m03; m13=m0.m13 - m1.m13; 
        m23=m0.m23 - m1.m23; m33=m0.m33 - m1.m33;       
    }

    void mul( const Matrix4f& m0, const Matrix4f& m1 ) {
        set(
            m0.m00*m1.m00 + m0.m01*m1.m10 + m0.m02*m1.m20 + m0.m03*m1.m30,
            m0.m10*m1.m00 + m0.m11*m1.m10 + m0.m12*m1.m20 + m0.m13*m1.m30,
            m0.m20*m1.m00 + m0.m21*m1.m10 + m0.m22*m1.m20 + m0.m23*m1.m30,
            m0.m30*m1.m00 + m0.m31*m1.m10 + m0.m32*m1.m20 + m0.m33*m1.m30,

            m0.m00*m1.m01 + m0.m01*m1.m11 + m0.m02*m1.m21 + m0.m03*m1.m31,
            m0.m10*m1.m01 + m0.m11*m1.m11 + m0.m12*m1.m21 + m0.m13*m1.m31,
            m0.m20*m1.m01 + m0.m21*m1.m11 + m0.m22*m1.m21 + m0.m23*m1.m31,
            m0.m30*m1.m01 + m0.m31*m1.m11 + m0.m32*m1.m21 + m0.m33*m1.m31,

            m0.m00*m1.m02 + m0.m01*m1.m12 + m0.m02*m1.m22 + m0.m03*m1.m32,
            m0.m10*m1.m02 + m0.m11*m1.m12 + m0.m12*m1.m22 + m0.m13*m1.m32,
            m0.m20*m1.m02 + m0.m21*m1.m12 + m0.m22*m1.m22 + m0.m23*m1.m32,
            m0.m30*m1.m02 + m0.m31*m1.m12 + m0.m32*m1.m22 + m0.m33*m1.m32,

            m0.m00*m1.m03 + m0.m01*m1.m13 + m0.m02*m1.m23 + m0.m03*m1.m33,
            m0.m10*m1.m03 + m0.m11*m1.m13 + m0.m12*m1.m23 + m0.m13*m1.m33,
            m0.m20*m1.m03 + m0.m21*m1.m13 + m0.m22*m1.m23 + m0.m23*m1.m33,
            m0.m30*m1.m03 + m0.m31*m1.m13 + m0.m32*m1.m23 + m0.m33*m1.m33
            );
    }

    void operator+=( const Matrix4f& m ) {
        m00+=m.m00; m10+=m.m10;
        m20+=m.m20; m30+=m.m30;
        m01+=m.m01; m11+=m.m11; 
        m21+=m.m21; m31+=m.m31;
        m02+=m.m02; m12+=m.m12; 
        m22+=m.m22; m32+=m.m32;
        m03+=m.m03; m13+=m.m13; 
        m23+=m.m23; m33+=m.m33;     
    }

    void operator-=( const Matrix4f& m ) {
        m00-=m.m00; m10-=m.m10;
        m20-=m.m20; m30-=m.m30;
        m01-=m.m01; m11-=m.m11; 
        m21-=m.m21; m31-=m.m31;
        m02-=m.m02; m12-=m.m12; 
        m22-=m.m22; m32-=m.m32;
        m03-=m.m03; m13-=m.m13; 
        m23-=m.m23; m33-=m.m33;
    }

    void operator*=( const Matrix4f& m ) {
        set(
            m00*m.m00 + m01*m.m10 + m02*m.m20 + m03*m.m30,
            m10*m.m00 + m11*m.m10 + m12*m.m20 + m13*m.m30,
            m20*m.m00 + m21*m.m10 + m22*m.m20 + m23*m.m30,
            m30*m.m00 + m31*m.m10 + m32*m.m20 + m33*m.m30,

            m00*m.m01 + m01*m.m11 + m02*m.m21 + m03*m.m31,
            m10*m.m01 + m11*m.m11 + m12*m.m21 + m13*m.m31,
            m20*m.m01 + m21*m.m11 + m22*m.m21 + m23*m.m31,
            m30*m.m01 + m31*m.m11 + m32*m.m21 + m33*m.m31,

            m00*m.m02 + m01*m.m12 + m02*m.m22 + m03*m.m32,
            m10*m.m02 + m11*m.m12 + m12*m.m22 + m13*m.m32,
            m20*m.m02 + m21*m.m12 + m22*m.m22 + m23*m.m32,
            m30*m.m02 + m31*m.m12 + m32*m.m22 + m33*m.m32,

            m00*m.m03 + m01*m.m13 + m02*m.m23 + m03*m.m33,
            m10*m.m03 + m11*m.m13 + m12*m.m23 + m13*m.m33,
            m20*m.m03 + m21*m.m13 + m22*m.m23 + m23*m.m33,
            m30*m.m03 + m31*m.m13 + m32*m.m23 + m33*m.m33
            );
    }

    void operator*=(float f) {
        m00 *= f;
        m10 *= f;
        m20 *= f;
        m30 *= f;
        m01 *= f;
        m11 *= f;
        m21 *= f;
        m31 *= f;
        m02 *= f;
        m12 *= f;
        m22 *= f;
        m32 *= f;
        m03 *= f;
        m13 *= f;
        m23 *= f;
        m33 *= f;
    }

    void transpose() {
        float tmp;
        tmp = m01;  m01 = m10;  m10 = tmp;
        tmp = m02;  m02 = m20;  m20 = tmp;
        tmp = m03;  m03 = m30;  m30 = tmp;
        tmp = m12;  m12 = m21;  m21 = tmp;
        tmp = m13;  m13 = m31;  m31 = tmp;
        tmp = m23;  m23 = m32;  m32 = tmp;
    }

    float determinant() const {
        return
             (m00*m11 - m01*m10) * (m22*m33 - m23*m32)
            -(m00*m12 - m02*m10) * (m21*m33 - m23*m31)
            +(m00*m13 - m03*m10) * (m21*m32 - m22*m31)
            +(m01*m12 - m02*m11) * (m20*m33 - m23*m30)
            -(m01*m13 - m03*m11) * (m20*m32 - m22*m30)
            +(m02*m13 - m03*m12) * (m20*m31 - m21*m30);
    }

    void invert(const Matrix4f& m) {
        float s = m.determinant();
        if (s == 0.0f) {
            printf("matrix was singular\n");
            return;
        }
        
        s = 1.0f/s;
        set(
            m.m11*(m.m22*m.m33 - m.m23*m.m32) + m.m12*(m.m23*m.m31 - m.m21*m.m33) + m.m13*(m.m21*m.m32 - m.m22*m.m31),
            m.m12*(m.m20*m.m33 - m.m23*m.m30) + m.m13*(m.m22*m.m30 - m.m20*m.m32) + m.m10*(m.m23*m.m32 - m.m22*m.m33),
            m.m13*(m.m20*m.m31 - m.m21*m.m30) + m.m10*(m.m21*m.m33 - m.m23*m.m31) + m.m11*(m.m23*m.m30 - m.m20*m.m33),
            m.m10*(m.m22*m.m31 - m.m21*m.m32) + m.m11*(m.m20*m.m32 - m.m22*m.m30) + m.m12*(m.m21*m.m30 - m.m20*m.m31),

            m.m21*(m.m02*m.m33 - m.m03*m.m32) + m.m22*(m.m03*m.m31 - m.m01*m.m33) + m.m23*(m.m01*m.m32 - m.m02*m.m31),
            m.m22*(m.m00*m.m33 - m.m03*m.m30) + m.m23*(m.m02*m.m30 - m.m00*m.m32) + m.m20*(m.m03*m.m32 - m.m02*m.m33),
            m.m23*(m.m00*m.m31 - m.m01*m.m30) + m.m20*(m.m01*m.m33 - m.m03*m.m31) + m.m21*(m.m03*m.m30 - m.m00*m.m33),
            m.m20*(m.m02*m.m31 - m.m01*m.m32) + m.m21*(m.m00*m.m32 - m.m02*m.m30) + m.m22*(m.m01*m.m30 - m.m00*m.m31),

            m.m31*(m.m02*m.m13 - m.m03*m.m12) + m.m32*(m.m03*m.m11 - m.m01*m.m13) + m.m33*(m.m01*m.m12 - m.m02*m.m11),
            m.m32*(m.m00*m.m13 - m.m03*m.m10) + m.m33*(m.m02*m.m10 - m.m00*m.m12) + m.m30*(m.m03*m.m12 - m.m02*m.m13),
            m.m33*(m.m00*m.m11 - m.m01*m.m10) + m.m30*(m.m01*m.m13 - m.m03*m.m11) + m.m31*(m.m03*m.m10 - m.m00*m.m13),
            m.m30*(m.m02*m.m11 - m.m01*m.m12) + m.m31*(m.m00*m.m12 - m.m02*m.m10) + m.m32*(m.m01*m.m10 - m.m00*m.m11),

            m.m01*(m.m13*m.m22 - m.m12*m.m23) + m.m02*(m.m11*m.m23 - m.m13*m.m21) + m.m03*(m.m12*m.m21 - m.m11*m.m22),
            m.m02*(m.m13*m.m20 - m.m10*m.m23) + m.m03*(m.m10*m.m22 - m.m12*m.m20) + m.m00*(m.m12*m.m23 - m.m13*m.m22),
            m.m03*(m.m11*m.m20 - m.m10*m.m21) + m.m00*(m.m13*m.m21 - m.m11*m.m23) + m.m01*(m.m10*m.m23 - m.m13*m.m20),
            m.m00*(m.m11*m.m22 - m.m12*m.m21) + m.m01*(m.m12*m.m20 - m.m10*m.m22) + m.m02*(m.m10*m.m21 - m.m11*m.m20)
            );

        (*this) *= s;
    }

    void invert() {
        float s = determinant();
        if (s == 0.0f) {
            printf("matrix was singular\n");
            return;
        }
        
        s = 1.0f/s;
        set(
            m11*(m22*m33 - m23*m32) + m12*(m23*m31 - m21*m33) + m13*(m21*m32 - m22*m31),
            m12*(m20*m33 - m23*m30) + m13*(m22*m30 - m20*m32) + m10*(m23*m32 - m22*m33),
            m13*(m20*m31 - m21*m30) + m10*(m21*m33 - m23*m31) + m11*(m23*m30 - m20*m33),
            m10*(m22*m31 - m21*m32) + m11*(m20*m32 - m22*m30) + m12*(m21*m30 - m20*m31),

            m21*(m02*m33 - m03*m32) + m22*(m03*m31 - m01*m33) + m23*(m01*m32 - m02*m31),
            m22*(m00*m33 - m03*m30) + m23*(m02*m30 - m00*m32) + m20*(m03*m32 - m02*m33),
            m23*(m00*m31 - m01*m30) + m20*(m01*m33 - m03*m31) + m21*(m03*m30 - m00*m33),
            m20*(m02*m31 - m01*m32) + m21*(m00*m32 - m02*m30) + m22*(m01*m30 - m00*m31),

            m31*(m02*m13 - m03*m12) + m32*(m03*m11 - m01*m13) + m33*(m01*m12 - m02*m11),
            m32*(m00*m13 - m03*m10) + m33*(m02*m10 - m00*m12) + m30*(m03*m12 - m02*m13),
            m33*(m00*m11 - m01*m10) + m30*(m01*m13 - m03*m11) + m31*(m03*m10 - m00*m13),
            m30*(m02*m11 - m01*m12) + m31*(m00*m12 - m02*m10) + m32*(m01*m10 - m00*m11),

            m01*(m13*m22 - m12*m23) + m02*(m11*m23 - m13*m21) + m03*(m12*m21 - m11*m22),
            m02*(m13*m20 - m10*m23) + m03*(m10*m22 - m12*m20) + m00*(m12*m23 - m13*m22),
            m03*(m11*m20 - m10*m21) + m00*(m13*m21 - m11*m23) + m01*(m10*m23 - m13*m20),
            m00*(m11*m22 - m12*m21) + m01*(m12*m20 - m10*m22) + m02*(m10*m21 - m11*m20)
            );

        (*this) *= s;
    }

    // invert calculation of rot + trans matrix.
    void invertRT() {
        invertR();
        float tx = -(m00 * m03 + m01 * m13 + m02 * m23);
        float ty = -(m10 * m03 + m11 * m13 + m12 * m23);
        float tz = -(m20 * m03 + m21 * m13 + m22 * m23);
        m03 = tx; m13 = ty; m23 = tz;
        // m30=m31=m32=0.0f; m33=1.0f;
    }
    void invertRT(const Matrix4f& m) {
        invertR(m);
        m03 = -(m00 * m.m03 + m01 * m.m13 + m02 * m.m23);
        m13 = -(m10 * m.m03 + m11 * m.m13 + m12 * m.m23);
        m23 = -(m20 * m.m03 + m21 * m.m13 + m22 * m.m23);
        m30 = m31 = m32 = 0.0f; 
        m33 = 1.0f;
    }

    // invert calculation for rotation matrix (without trans)
    void invertR() {
        float tmp;
        tmp = m01; m01 = m10; m10 = tmp;
        tmp = m02; m02 = m20; m20 = tmp;
        tmp = m12; m12 = m21; m21 = tmp;
        // m03 = m13 = m23 = 0.0f;
        // m30 = m31 = m32 = 0.0f; 
        // m33 = 1.0f;
    }
    void invertR(const Matrix4f& m) {
        m00 = m.m00; m11 = m.m11; m22 = m.m22;
        m01 = m.m10; m10 = m.m01;
        m02 = m.m20; m20 = m.m02;
        m12 = m.m21; m21 = m.m12;
        m03 = m13 = m23 = 0.0f;
        m30 = m31 = m32 = 0.0f; 
        m33 = 1.0f;
    }

    void transform(const Vector4f& in, Vector4f& out) const {
        out.set(
            m00 * in.x + m01 * in.y + m02 * in.z + m03 * in.w,
            m10 * in.x + m11 * in.y + m12 * in.z + m13 * in.w,
            m20 * in.x + m21 * in.y + m22 * in.z + m23 * in.w,
            m30 * in.x + m31 * in.y + m32 * in.z + m33 * in.w
            );
    }

    void setRotationX(float angle) {
        float s = sinf(angle);
        float c = cosf(angle);

        set( 1.0f, 0.0f, 0.0f, 0.0f,
             0.0f,    c,    s, 0.0f,
             0.0f,   -s,    c ,0.0f,
             0.0f, 0.0f, 0.0f, 1.0f );
    }
    
    void setRotationY(float angle) {
        float s = sinf(angle);
        float c = cosf(angle);

        set(    c, 0.0f,   -s, 0.0f,
             0.0f, 1.0f, 0.0f, 0.0f,
                s, 0.0f,    c, 0.0f,
             0.0f, 0.0f, 0.0f, 1.0f );
    }

    void setRotationZ(float angle) {
        float s = sinf(angle);
        float c = cosf(angle);

        set(    c,    s, 0.0f, 0.0f,
               -s,    c, 0.0f, 0.0f,
             0.0f, 0.0f, 1.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 1.0f );
    }

    const float* getPointer() const {
        return reinterpret_cast<const float*>(this);
    }

    float* getPointer() {
        return reinterpret_cast<float*>(this);
    }

    void debugDump() const {
        printf( "[ [%f, %f, %f, %f ]\n"
                "  [%f, %f, %f, %f ]\n"
                "  [%f, %f, %f, %f ]\n"
                "  [%f, %f, %f, %f ] ] \n", 
                m00,m01,m02,m03,
                m10,m11,m12,m13,
                m20,m21,m22,m23,
                m30,m31,m32,m33 );
    }
} __attribute__ ( (aligned(16)) );


//====================================
//            [Quat4f]
//====================================
class Quat4f : public Vector4f {
public:
    Quat4f() : Vector4f() {}

    Quat4f(const Vector4f& v) 
        :
        Vector4f(v) {}

    Quat4f(float x, float y, float z, float w) 
        :
        Vector4f(x, y, z, w) {}
    
    void set(float x, float y, float z, float w) {
        Vector4f::set(x, y, z, w);
    }

    void mul(const Quat4f& q0, const Quat4f& q1) {
        float x0, y0, z0, w0;
        float x1, y1, z1, w1;

        x0 = q0.x;
        y0 = q0.y;
        z0 = q0.z;
        w0 = q0.w;

        x1 = q1.x;
        y1 = q1.y;
        z1 = q1.z;
        w1 = q1.w;

        Vector4f::set( x0*w1 + w0*x1 + y0*z1 - z0*y1,
                       y0*w1 + w0*y1 + z0*x1 - x0*z1,
                       z0*w1 + w0*z1 + x0*y1 - y0*x1,
                       w0*w1 - x0*x1 - y0*y1 - z0*z1 );
    }

    void operator*=(const Quat4f& q) {
        float x1 = q.x;
        float y1 = q.y;
        float z1 = q.z;
        float w1 = q.w;

        Vector4f::set( x*w1 + w*x1 + y*z1 - z*y1,
                       y*w1 + w*y1 + z*x1 - x*z1,
                       z*w1 + w*z1 + x*y1 - y*x1,
                       w*w1 - x*x1 - y*y1 - z*z1 );     
    }

    void div(float f, const Quat4f& q) {
        float x1 = q.x;
        float y1 = q.y;
        float z1 = q.z;
        float w1 = q.w;
        // Zero division may occur.
        float n = f / q.lengthSquared();
        Vector4f::set( -x1 * n, -y1 * n, -z1 * n, w1 * n );
    }

    void set(const Matrix4f& m) {
        // Absolute deteminant of m
        const float dt = fabsf( m.determinant() );
        // m without scale.
        Matrix4f o(m);
        o *= (1.0f/dt);
        float tr = o.m00 + o.m11 + o.m22; // trace of o
        float f0;

        if( tr >= 0.0f ) {
            f0 = sqrtf( tr + 1.0f );
            w = 0.5f * f0;
            f0 = 0.5f / f0;
            x = ( o.m21 - o.m12 ) * f0;
            y = ( o.m02 - o.m20 ) * f0;
            z = ( o.m10 - o.m01 ) * f0;
        } else {
            int i;
            float max;
            i = 0;
            // Which axis x, y, z is nearest to rotation axis.
            max = o.m00;
            if( o.m11 > max ) {
                i = 1;
                max = o.m11;
            }
            if( o.m22 > max ) {
                i = 2;
            }
            switch(i) {
            case 0 :
                // axis x is nearest
                f0 = sqrtf( o.m00 - o.m11 - o.m22 + 1.0f );
                x  = 0.5f * f0;
                f0 = 0.5f / f0;
                y = ( o.m01 + o.m10 ) * f0;
                z = ( o.m20 + o.m02 ) * f0;
                w = ( o.m21 - o.m12 ) * f0;
                break;
            case 1 :
                // axis y is nearest
                f0 = sqrtf( o.m11 - o.m22 - o.m00 + 1.0f );
                y  = 0.5f * f0;
                f0 = 0.5f / f0;
                z = ( o.m12 + o.m21 ) * f0;
                x = ( o.m01 + o.m10 ) * f0;
                w = ( o.m02 - o.m20 ) * f0;
                break;
            case 2 :
                // axis z is nearest
                f0 = sqrtf( o.m22 - o.m00 - o.m11 + 1.0f );
                z  = 0.5f * f0;
                f0 = 0.5f / f0;
                x = ( o.m20 + o.m02 ) * f0;
                y = ( o.m12 + o.m21 ) * f0;
                w = ( o.m10 - o.m01 ) * f0;
                break;
            }
        }
    }
} __attribute__ ( (aligned(16)) );

/**
 * Matrix4f::set():
 */
inline void Matrix4f::set(const Quat4f& q) {
    Quat4f s;
    s.div(1.0f, q);

    Quat4f qx(1.0f, 0.0f, 0.0f, 0.0f);
    Quat4f qy(0.0f, 1.0f, 0.0f, 0.0f);
    Quat4f qz(0.0f, 0.0f, 1.0f, 0.0f);

    qx.mul(qx, s); qx.mul(q, qx);
    qy.mul(qy, s); qy.mul(q, qy);
    qz.mul(qz, s); qz.mul(q, qz);

    setColumn(0 , qx);
    setColumn(1 , qy);
    setColumn(2 , qz);
    
    const Vector4f vw(0.0f, 0.0f, 0.0f, 1.0f);
    setColumn(3, vw);
}

#endif
