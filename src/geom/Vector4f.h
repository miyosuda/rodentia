// -*- C++ -*-
#ifndef VECTOR4F_HEADER
#define VECTOR4F_HEADER

#include <math.h>
#include <stdio.h>

class Vector4f {
public:
    float x;
    float y;
    float z;
    float w;
    
    Vector4f() {}

    Vector4f(float x, float y, float z, float w) {
        set(x, y, z, w);
    }

    Vector4f(const Vector4f& v){
        set(v);
    }

    void set(const Vector4f& v) {
        x = v.x; y = v.y; z = v.z; w = v.w;
    }

    void set(float x_, float y_, float z_, float w_) {
        x = x_; y = y_; z = z_; w = w_;
    }
    
    void setZero() {
        x = y = z = w = 0.0f;
    }

    void add(const Vector4f& v0, const Vector4f& v1) {
        x = v0.x + v1.x;
        y = v0.y + v1.y;
        z = v0.z + v1.z;
        w = v0.w + v1.w;
    }

    void sub(const Vector4f& v0, const Vector4f& v1) {
        x = v0.x - v1.x;
        y = v0.y - v1.y;
        z = v0.z - v1.z;
        w = v0.w - v1.w;
    }

    void scale(float rate, 
               const Vector4f& v) {
        x = rate * v.x;
        y = rate * v.y;
        z = rate * v.z; 
        w = rate * v.w;
    }

    void scaleAdd(float rate, 
                  const Vector4f& v0, 
                  const Vector4f& v1) {
        x = rate * v0.x + v1.x;
        y = rate * v0.y + v1.y;
        z = rate * v0.z + v1.z;
        w = rate * v0.w + v1.w;
    }

    void scaleAdd(float rate, 
                  const Vector4f& v) {
        x += rate * v.x;
        y += rate * v.y;
        z += rate * v.z;
        w += rate * v.w;
    }

    float lengthSquared() const {
        return x*x + y*y + z*z + w*w;
    }

    float length() const {
        return sqrtf( lengthSquared() );
    }   

    float dot(const Vector4f& v) const {
        return x*v.x + y*v.y + z*v.z + w*v.w;
    }

    // 3-dim cross
    void cross(const Vector4f& v0, const Vector4f& v1) {
        set( v0.y * v1.z - v0.z * v1.y,
             v0.z * v1.x - v0.x * v1.z,
             v0.x * v1.y - v0.y * v1.x,
             0.0f );
    }
    
    void normalize() {
        float d = length();
        // Zero division may occur
        x /= d;
        y /= d;
        z /= d;
        w /= d;
    }

    Vector4f& operator=(const Vector4f& v) {
        set(v);
        return *this;
    }

    void operator+=(const Vector4f& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
    }

    void operator-=(const Vector4f& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
    }

    void operator*=(float f) {
        x *= f;
        y *= f;
        z *= f;
        w *= f;
    }

    void operator/=(float f) {
        // ゼロ除算の可能性あり.
        x /= f;
        y /= f;
        z /= f;
        w /= f;
    }

    void parallelProduct(const Vector4f& v) {
        x = x * v.x;
        y = y * v.y;
        z = z * v.z;
        w = w * v.w;
    }

    void parallelProduct(const Vector4f& v0, const Vector4f& v1) {
        x = v0.x * v1.x;
        y = v0.y * v1.y;
        z = v0.z * v1.z;
        w = v0.w * v1.w;        
    }

    float get(unsigned int i) const {
        return ( ( static_cast<const float*>(&x) )[i] );
    }

    const float* getPointer() const {
        return reinterpret_cast<const float*>(this);
    }

    const int* getIntPointer() const {
        return reinterpret_cast<const int*>(this);
    }

    float* getPointer() {
        return reinterpret_cast<float*>(this);
    }   

    int* getIntPointer() {
        return reinterpret_cast<int*>(this);
    }

    void debugDump() const {
        printf("( %f, %f, %f, %f )\n", x, y, z, w);     
    }
} __attribute__ ( (aligned(16)) ) ;

#endif


