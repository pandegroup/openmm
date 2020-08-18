#ifndef OPENMM_COMPUTEVECTORTYPES_H_
#define OPENMM_COMPUTEVECTORTYPES_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2019 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

namespace OpenMM {

struct mm_short2 {
    short x, y;
    mm_short2() {
    }
    mm_short2(short x, short y) : x(x), y(y) {
    }
};
struct mm_short3 {
    short x, y, z, w;
    mm_short3() {
    }
    mm_short3(short x, short y, short z) : x(x), y(y), z(z) {
    }
};
struct mm_short4 {
    short x, y, z, w;
    mm_short4() {
    }
    mm_short4(short x, short y, short z, short w) : x(x), y(y), z(z), w(w) {
    }
};
struct mm_int2 {
    int x, y;
    mm_int2() {
    }
    mm_int2(int x, int y) : x(x), y(y) {
    }
};
struct mm_int3 {
    int x, y, z, w;
    mm_int3() {
    }
    mm_int3(int x, int y, int z) : x(x), y(y), z(z) {
    }
};
struct mm_int4 {
    int x, y, z, w;
    mm_int4() {
    }
    mm_int4(int x, int y, int z, int w) : x(x), y(y), z(z), w(w) {
    }
};
struct mm_long4 {
    long x, y, z, w;
    mm_long4() {
    }
    mm_long4(long x, long y, long z, long w) : x(x), y(y), z(z), w(w) {
    }
};
struct mm_float2 {
    float x, y;
    mm_float2() {
    }
    mm_float2(float x, float y) : x(x), y(y) {
    }
};
struct mm_float3 {
    float x, y, z, w;
    mm_float3() {
    }
    mm_float3(float x, float y, float z) : x(x), y(y), z(z) {
    }
};
struct mm_float4 {
    float x, y, z, w;
    mm_float4() {
    }
    mm_float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {
    }
};
struct mm_double2 {
    double x, y;
    mm_double2() {
    }
    mm_double2(double x, double y) : x(x), y(y) {
    }
};
struct mm_double3 {
    double x, y, z, w;
    mm_double3() {
    }
    mm_double3(double x, double y, double z) : x(x), y(y), z(z) {
    }
};
struct mm_double4 {
    double x, y, z, w;
    mm_double4() {
    }
    mm_double4(double x, double y, double z, double w) : x(x), y(y), z(z), w(w) {
    }
};

} // namespace OpenMM

#endif /*OPENMM_COMPUTEVECTORTYPES_H_*/
