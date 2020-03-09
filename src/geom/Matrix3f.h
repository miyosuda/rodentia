// -*- C++ -*-
#ifndef MATRIX3F_HEADER
#define MATRIX3F_HEADER

/**
 * Member order is
 *    [m00, m10, ....]
 * Column major order.

 * +-------------+
 * |m00, m01, m02|
 * |m10, m11, m12|
 * |m20, m21, m22|
 * +-------------+
 */
class Matrix3f {
public:
    float m00; // row1 column1
    float m10; // row2 column1
    float m20; // row3 column1
    
    float m01;
    float m11;
    float m21;
    
    float m02;
    float m12;
    float m22;
    
    Matrix3f() {}
    
    /**
     * convert Matrix4f -> Matrix3f
     */
    void set(const Matrix4f& m) {
        m00=m.m00; m10=m.m10; m20=m.m20;
        m01=m.m01; m11=m.m11; m21=m.m21;
        m02=m.m02; m12=m.m12; m22=m.m22;
    }
    
    void setIdentity() {
        m10=m20 = 0.0f;
        m01=m21 = 0.0f;
        m02=m12 = 0.0f;
        m00 = m11 = m22 = 1.0f;
    }

    void set(
        float m00_, float m10_, float m20_,
        float m01_, float m11_, float m21_,
        float m02_, float m12_, float m22_
        ) {
        m00=m00_; m10=m10_; m20=m20_;
        m01=m01_; m11=m11_; m21=m21_;
        m02=m02_; m12=m12_; m22=m22_;
    }
    
    void operator*=(float scalar) {
        m00 *= scalar; m01 *= scalar;  m02 *= scalar;
        m10 *= scalar; m11 *= scalar;  m12 *= scalar;
        m20 *= scalar; m21 *= scalar;  m22 *= scalar;
     }
        
    void transpose() {
        float tmp;
        tmp = m01;  m01 = m10;  m10 = tmp;
        tmp = m02;  m02 = m20;  m20 = tmp;
        tmp = m12;  m12 = m21;  m21 = tmp;
    }

    void invertR() {
        float s = determinant();
        if (s == 0.0f) {
            return;
        }
        
        transpose();
        (*this) *= s;
    }
    
    float determinant() const {
        return m00*(m11*m22 - m21*m12)
            - m01*(m10*m22 - m20*m12)
            + m02*(m10*m21 - m20*m11);
    }
    
    const float* getPointer() const {
        return reinterpret_cast<const float*>(this);
    }
    
    float* getPointer() {
        return reinterpret_cast<float*>(this);
    }
};

#endif
