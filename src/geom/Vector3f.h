#ifndef VECOT3F_H
#define VECOT3F_H

#include <math.h>
#include <stdio.h>

class Vector3f {
public:
    float x;
    float y;
    float z;
    
    Vector3f() {}
    
    Vector3f(const float * v){
        set(v[0],v[1],v[2]);
    }
    
    Vector3f(float x, float y, float z) {
        set(x, y, z);
    }
    
    Vector3f(const Vector3f& v){
        set(v);
    }
    
    void set(const Vector3f& v) {
        x = v.x; y = v.y; z = v.z; 
    }
    
    void set(float x_, float y_, float z_) {
        x = x_; y = y_; z = z_;
    }
    
    void setZero() {
        x = y = z = 0.0f;
    }
    
    void add(const Vector3f& v0, const Vector3f& v1) {
        x = v0.x + v1.x;
        y = v0.y + v1.y;
        z = v0.z + v1.z;
    }
    
    void sub(const Vector3f& v0, const Vector3f& v1) {
        x = v0.x - v1.x;
        y = v0.y - v1.y;
        z = v0.z - v1.z;
    }
    
    void scale(float rate, 
               const Vector3f& v) {
        x = rate * v.x;
        y = rate * v.y;
        z = rate * v.z; 
    }
    
    void scaleAdd(float rate, 
                  const Vector3f& v0, 
                  const Vector3f& v1) {
        x = rate * v0.x + v1.x;
        y = rate * v0.y + v1.y;
        z = rate * v0.z + v1.z;
    }
    
    void scaleAdd(float rate, 
                  const Vector3f& v) {
        x += rate * v.x;
        y += rate * v.y;
        z += rate * v.z;
    }
    
    float lengthSquared() const {
        return x*x + y*y + z*z;
    }
    
    float length() const {
        return sqrtf( lengthSquared() );
    }   
    
    float dot(const Vector3f& v) const {
        return x*v.x + y*v.y + z*v.z;
    }
    
    void cross(const Vector3f& v0, const Vector3f& v1) {
        set( v0.y * v1.z - v0.z * v1.y,
             v0.z * v1.x - v0.x * v1.z,
             v0.x * v1.y - v0.y * v1.x);
    }
    
    void normalize() {
        float d = length();
        // Zero division may occur
        x /= d;
        y /= d;
        z /= d;
    }
    
    Vector3f& operator=(const Vector3f& v) {
        set(v);
        return *this;
    }
    
    void operator+=(const Vector3f& v) {
        x += v.x;
        y += v.y;
        z += v.z;
    }
    
    void operator-=(const Vector3f& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
    }
    
    void operator*=(float f) {
        x *= f;
        y *= f;
        z *= f;
    }
    
    void operator/=(float f) {
        // Zero division may occur
        x /= f;
        y /= f;
        z /= f;
    }
    
    void parallelProduct(const Vector3f& v) {
        x = x * v.x;
        y = y * v.y;
        z = z * v.z;
    }
    
    void parallelProduct(const Vector3f& v0, const Vector3f& v1) {
        x = v0.x * v1.x;
        y = v0.y * v1.y;
        z = v0.z * v1.z;
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
        printf("( %f, %f, %f)\n", x, y, z);
    }
};  

#endif
