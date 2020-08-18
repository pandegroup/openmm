/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2015 Stanford University and the Authors.      *
 * Portions copyright (C) 2020 Advanced Micro Devices, Inc. All Rights        *
 * Reserved.                                                                  *
 * Authors: Peter Eastman, Nicholas Curtis                                    *
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

#include "HipFFT3D.h"
#include "HipContext.h"
#include "HipKernelSources.h"
#include "SimTKOpenMMRealType.h"
#include <map>
#include <sstream>
#include <string>

using namespace OpenMM;
using namespace std;

HipFFT3D::HipFFT3D(HipContext& context, int xsize, int ysize, int zsize, bool realToComplex) :
        context(context), xsize(xsize), ysize(ysize), zsize(zsize) {
    packRealAsComplex = false;
    int packedXSize = xsize;
    int packedYSize = ysize;
    int packedZSize = zsize;
    if (realToComplex) {
        // If any axis size is even, we can pack the real values into a complex grid that is only half as large.
        // Look for an appropriate axis.

        packRealAsComplex = true;
        int packedAxis, bufferSize;
        if (xsize%2 == 0) {
            packedAxis = 0;
            packedXSize /= 2;
            bufferSize = packedXSize;
        }
        else if (ysize%2 == 0) {
            packedAxis = 1;
            packedYSize /= 2;
            bufferSize = packedYSize;
        }
        else if (zsize%2 == 0) {
            packedAxis = 2;
            packedZSize /= 2;
            bufferSize = packedZSize;
        }
        else
            packRealAsComplex = false;
        if (packRealAsComplex) {
            // Build the kernels for packing and unpacking the data.

            map<string, string> defines;
            defines["XSIZE"] = context.intToString(xsize);
            defines["YSIZE"] = context.intToString(ysize);
            defines["ZSIZE"] = context.intToString(zsize);
            defines["PACKED_AXIS"] = context.intToString(packedAxis);
            defines["PACKED_XSIZE"] = context.intToString(packedXSize);
            defines["PACKED_YSIZE"] = context.intToString(packedYSize);
            defines["PACKED_ZSIZE"] = context.intToString(packedZSize);
            defines["M_PI"] = context.doubleToString(M_PI);
            hipModule_t module = context.createModule(HipKernelSources::vectorOps+HipKernelSources::fftR2C, defines);
            packForwardKernel = context.getKernel(module, "packForwardData");
            unpackForwardKernel = context.getKernel(module, "unpackForwardData");
            packBackwardKernel = context.getKernel(module, "packBackwardData");
            unpackBackwardKernel = context.getKernel(module, "unpackBackwardData");
        }
    }
    bool inputIsReal = (realToComplex && !packRealAsComplex);
    zkernel = createKernel(packedXSize, packedYSize, packedZSize, zthreads, 0, true, inputIsReal);
    xkernel = createKernel(packedYSize, packedZSize, packedXSize, xthreads, 1, true, inputIsReal);
    ykernel = createKernel(packedZSize, packedXSize, packedYSize, ythreads, 2, true, inputIsReal);
    invzkernel = createKernel(packedXSize, packedYSize, packedZSize, zthreads, 0, false, inputIsReal);
    invxkernel = createKernel(packedYSize, packedZSize, packedXSize, xthreads, 1, false, inputIsReal);
    invykernel = createKernel(packedZSize, packedXSize, packedYSize, ythreads, 2, false, inputIsReal);
}

void HipFFT3D::execFFT(HipArray& in, HipArray& out, bool forward) {
    hipFunction_t kernel1 = (forward ? zkernel : invzkernel);
    hipFunction_t kernel2 = (forward ? xkernel : invxkernel);
    hipFunction_t kernel3 = (forward ? ykernel : invykernel);
    void* args1[] = {&in.getDevicePointer(), &out.getDevicePointer()};
    void* args2[] = {&out.getDevicePointer(), &in.getDevicePointer()};
    if (packRealAsComplex) {
        hipFunction_t packKernel = (forward ? packForwardKernel : packBackwardKernel);
        hipFunction_t unpackKernel = (forward ? unpackForwardKernel : unpackBackwardKernel);
        int gridSize = xsize*ysize*zsize/2;

        // Pack the data into a half sized grid.

        context.executeKernel(packKernel, args1, gridSize, 128);

        // Perform the FFT.

        context.executeKernel(kernel1, args2, gridSize, zthreads);
        context.executeKernel(kernel2, args1, gridSize, xthreads);
        context.executeKernel(kernel3, args2, gridSize, ythreads);

        // Unpack the data.

        context.executeKernel(unpackKernel, args1, gridSize, 128);
    }
    else {
        context.executeKernel(kernel1, args1, xsize*ysize*zsize, zthreads);
        context.executeKernel(kernel2, args2, xsize*ysize*zsize, xthreads);
        context.executeKernel(kernel3, args1, xsize*ysize*zsize, ythreads);
    }
}

int HipFFT3D::findLegalDimension(int minimum) {
    if (minimum < 1)
        return 1;
    while (true) {
        // Attempt to factor the current value.

        int unfactored = minimum;
        for (int factor = 2; factor < 8; factor++) {
            while (unfactored > 1 && unfactored%factor == 0)
                unfactored /= factor;
        }
        if (unfactored == 1)
            return minimum;
        minimum++;
    }
}


static int getSmallestRadix(int size) {
    int minRadix = 1;
    int unfactored = size;
    while (unfactored%7 == 0) {
        minRadix = 7;
        unfactored /= 7;
    }
    while (unfactored%5 == 0) {
        minRadix = 5;
        unfactored /= 5;
    }
    while (unfactored%4 == 0) {
        minRadix = 4;
        unfactored /= 4;
    }
    while (unfactored%3 == 0) {
        minRadix = 3;
        unfactored /= 3;
    }
    while (unfactored%2 == 0) {
        minRadix = 2;
        unfactored /= 2;
    }
    return minRadix;
}

hipFunction_t HipFFT3D::createKernel(int xsize, int ysize, int zsize, int& threads, int axis, bool forward, bool inputIsReal) {
    int maxThreads = (context.getUseDoublePrecision() ? 128 : 256);
//    while (maxThreads > 128 && maxThreads-64 >= zsize)
//        maxThreads -= 64;
    int threadsPerBlock = zsize/getSmallestRadix(zsize);
    stringstream source;
    int blocksPerGroup = max(1, maxThreads/threadsPerBlock);
    int stage = 0;
    int L = zsize;
    int m = 1;

    // Factor zsize, generating an appropriate block of code for each factor.

    while (L > 1) {
        int input = stage%2;
        int output = 1-input;
        int radix;
        if (L%7 == 0)
            radix = 7;
        else if (L%5 == 0)
            radix = 5;
        else if (L%4 == 0)
            radix = 4;
        else if (L%3 == 0)
            radix = 3;
        else if (L%2 == 0)
            radix = 2;
        else
            throw OpenMMException("Illegal size for FFT: "+context.intToString(zsize));
        source<<"{\n";
        L = L/radix;
        source<<"// Pass "<<(stage+1)<<" (radix "<<radix<<")\n";
        if (L*m < threadsPerBlock)
            source<<"if (threadIdx.x < "<<(blocksPerGroup*L*m)<<") {\n";
        else
            source<<"{\n";
        source<<"int block = threadIdx.x/"<<(L*m)<<";\n";
        source<<"int i = threadIdx.x-block*"<<(L*m)<<";\n";
        source<<"int base = i+block*"<<zsize<<";\n";
        source<<"int j = i/"<<m<<";\n";
        if (radix == 7) {
            source<<"real2 c0 = data"<<input<<"[base];\n";
            source<<"real2 c1 = data"<<input<<"[base+"<<(L*m)<<"];\n";
            source<<"real2 c2 = data"<<input<<"[base+"<<(2*L*m)<<"];\n";
            source<<"real2 c3 = data"<<input<<"[base+"<<(3*L*m)<<"];\n";
            source<<"real2 c4 = data"<<input<<"[base+"<<(4*L*m)<<"];\n";
            source<<"real2 c5 = data"<<input<<"[base+"<<(5*L*m)<<"];\n";
            source<<"real2 c6 = data"<<input<<"[base+"<<(6*L*m)<<"];\n";
            source<<"real2 d0 = c1+c6;\n";
            source<<"real2 d1 = c1-c6;\n";
            source<<"real2 d2 = c2+c5;\n";
            source<<"real2 d3 = c2-c5;\n";
            source<<"real2 d4 = c4+c3;\n";
            source<<"real2 d5 = c4-c3;\n";
            source<<"real2 d6 = d2+d0;\n";
            source<<"real2 d7 = d5+d3;\n";
            source<<"real2 b0 = c0+d6+d4;\n";
            source<<"real2 b1 = "<<context.doubleToString((cos(2*M_PI/7)+cos(4*M_PI/7)+cos(6*M_PI/7))/3-1)<<"*(d6+d4);\n";
            source<<"real2 b2 = "<<context.doubleToString((2*cos(2*M_PI/7)-cos(4*M_PI/7)-cos(6*M_PI/7))/3)<<"*(d0-d4);\n";
            source<<"real2 b3 = "<<context.doubleToString((cos(2*M_PI/7)-2*cos(4*M_PI/7)+cos(6*M_PI/7))/3)<<"*(d4-d2);\n";
            source<<"real2 b4 = "<<context.doubleToString((cos(2*M_PI/7)+cos(4*M_PI/7)-2*cos(6*M_PI/7))/3)<<"*(d2-d0);\n";
            source<<"real2 b5 = -(SIGN)*"<<context.doubleToString((sin(2*M_PI/7)+sin(4*M_PI/7)-sin(6*M_PI/7))/3)<<"*(d7+d1);\n";
            source<<"real2 b6 = -(SIGN)*"<<context.doubleToString((2*sin(2*M_PI/7)-sin(4*M_PI/7)+sin(6*M_PI/7))/3)<<"*(d1-d5);\n";
            source<<"real2 b7 = -(SIGN)*"<<context.doubleToString((sin(2*M_PI/7)-2*sin(4*M_PI/7)-sin(6*M_PI/7))/3)<<"*(d5-d3);\n";
            source<<"real2 b8 = -(SIGN)*"<<context.doubleToString((sin(2*M_PI/7)+sin(4*M_PI/7)+2*sin(6*M_PI/7))/3)<<"*(d3-d1);\n";
            source<<"real2 t0 = b0+b1;\n";
            source<<"real2 t1 = b2+b3;\n";
            source<<"real2 t2 = b4-b3;\n";
            source<<"real2 t3 = -b2-b4;\n";
            source<<"real2 t4 = b6+b7;\n";
            source<<"real2 t5 = b8-b7;\n";
            source<<"real2 t6 = -b8-b6;\n";
            source<<"real2 t7 = t0+t1;\n";
            source<<"real2 t8 = t0+t2;\n";
            source<<"real2 t9 = t0+t3;\n";
            source<<"real2 t10 = make_real2(t4.y+b5.y, -(t4.x+b5.x));\n";
            source<<"real2 t11 = make_real2(t5.y+b5.y, -(t5.x+b5.x));\n";
            source<<"real2 t12 = make_real2(t6.y+b5.y, -(t6.x+b5.x));\n";
            source<<"data"<<output<<"[base+6*j*"<<m<<"] = b0;\n";
            source<<"data"<<output<<"[base+(6*j+1)*"<<m<<"] = multiplyComplex(w[j*"<<zsize<<"/"<<(7*L)<<"], t7-t10);\n";
            source<<"data"<<output<<"[base+(6*j+2)*"<<m<<"] = multiplyComplex(w[j*"<<(2*zsize)<<"/"<<(7*L)<<"], t9-t12);\n";
            source<<"data"<<output<<"[base+(6*j+3)*"<<m<<"] = multiplyComplex(w[j*"<<(3*zsize)<<"/"<<(7*L)<<"], t8+t11);\n";
            source<<"data"<<output<<"[base+(6*j+4)*"<<m<<"] = multiplyComplex(w[j*"<<(4*zsize)<<"/"<<(7*L)<<"], t8-t11);\n";
            source<<"data"<<output<<"[base+(6*j+5)*"<<m<<"] = multiplyComplex(w[j*"<<(5*zsize)<<"/"<<(7*L)<<"], t9+t12);\n";
            source<<"data"<<output<<"[base+(6*j+6)*"<<m<<"] = multiplyComplex(w[j*"<<(6*zsize)<<"/"<<(7*L)<<"], t7+t10);\n";
        }
        else if (radix == 5) {
            source<<"real2 c0 = data"<<input<<"[base];\n";
            source<<"real2 c1 = data"<<input<<"[base+"<<(L*m)<<"];\n";
            source<<"real2 c2 = data"<<input<<"[base+"<<(2*L*m)<<"];\n";
            source<<"real2 c3 = data"<<input<<"[base+"<<(3*L*m)<<"];\n";
            source<<"real2 c4 = data"<<input<<"[base+"<<(4*L*m)<<"];\n";
            source<<"real2 d0 = c1+c4;\n";
            source<<"real2 d1 = c2+c3;\n";
            source<<"real2 d2 = "<<context.doubleToString(sin(0.4*M_PI))<<"*(c1-c4);\n";
            source<<"real2 d3 = "<<context.doubleToString(sin(0.4*M_PI))<<"*(c2-c3);\n";
            source<<"real2 d4 = d0+d1;\n";
            source<<"real2 d5 = "<<context.doubleToString(0.25*sqrt(5.0))<<"*(d0-d1);\n";
            source<<"real2 d6 = c0-0.25f*d4;\n";
            source<<"real2 d7 = d6+d5;\n";
            source<<"real2 d8 = d6-d5;\n";
            string coeff = context.doubleToString(sin(0.2*M_PI)/sin(0.4*M_PI));
            source<<"real2 d9 = (SIGN)*make_real2(d2.y+"<<coeff<<"*d3.y, -d2.x-"<<coeff<<"*d3.x);\n";
            source<<"real2 d10 = (SIGN)*make_real2("<<coeff<<"*d2.y-d3.y, d3.x-"<<coeff<<"*d2.x);\n";
            source<<"data"<<output<<"[base+4*j*"<<m<<"] = c0+d4;\n";
            source<<"data"<<output<<"[base+(4*j+1)*"<<m<<"] = multiplyComplex(w[j*"<<zsize<<"/"<<(5*L)<<"], d7+d9);\n";
            source<<"data"<<output<<"[base+(4*j+2)*"<<m<<"] = multiplyComplex(w[j*"<<(2*zsize)<<"/"<<(5*L)<<"], d8+d10);\n";
            source<<"data"<<output<<"[base+(4*j+3)*"<<m<<"] = multiplyComplex(w[j*"<<(3*zsize)<<"/"<<(5*L)<<"], d8-d10);\n";
            source<<"data"<<output<<"[base+(4*j+4)*"<<m<<"] = multiplyComplex(w[j*"<<(4*zsize)<<"/"<<(5*L)<<"], d7-d9);\n";
        }
        else if (radix == 4) {
            source<<"real2 c0 = data"<<input<<"[base];\n";
            source<<"real2 c1 = data"<<input<<"[base+"<<(L*m)<<"];\n";
            source<<"real2 c2 = data"<<input<<"[base+"<<(2*L*m)<<"];\n";
            source<<"real2 c3 = data"<<input<<"[base+"<<(3*L*m)<<"];\n";
            source<<"real2 d0 = c0+c2;\n";
            source<<"real2 d1 = c0-c2;\n";
            source<<"real2 d2 = c1+c3;\n";
            source<<"real2 d3 = (SIGN)*make_real2(c1.y-c3.y, c3.x-c1.x);\n";
            source<<"data"<<output<<"[base+3*j*"<<m<<"] = d0+d2;\n";
            source<<"data"<<output<<"[base+(3*j+1)*"<<m<<"] = multiplyComplex(w[j*"<<zsize<<"/"<<(4*L)<<"], d1+d3);\n";
            source<<"data"<<output<<"[base+(3*j+2)*"<<m<<"] = multiplyComplex(w[j*"<<(2*zsize)<<"/"<<(4*L)<<"], d0-d2);\n";
            source<<"data"<<output<<"[base+(3*j+3)*"<<m<<"] = multiplyComplex(w[j*"<<(3*zsize)<<"/"<<(4*L)<<"], d1-d3);\n";
        }
        else if (radix == 3) {
            source<<"real2 c0 = data"<<input<<"[base];\n";
            source<<"real2 c1 = data"<<input<<"[base+"<<(L*m)<<"];\n";
            source<<"real2 c2 = data"<<input<<"[base+"<<(2*L*m)<<"];\n";
            source<<"real2 d0 = c1+c2;\n";
            source<<"real2 d1 = c0-0.5f*d0;\n";
            source<<"real2 d2 = (SIGN)*"<<context.doubleToString(sin(M_PI/3.0))<<"*make_real2(c1.y-c2.y, c2.x-c1.x);\n";
            source<<"data"<<output<<"[base+2*j*"<<m<<"] = c0+d0;\n";
            source<<"data"<<output<<"[base+(2*j+1)*"<<m<<"] = multiplyComplex(w[j*"<<zsize<<"/"<<(3*L)<<"], d1+d2);\n";
            source<<"data"<<output<<"[base+(2*j+2)*"<<m<<"] = multiplyComplex(w[j*"<<(2*zsize)<<"/"<<(3*L)<<"], d1-d2);\n";
        }
        else if (radix == 2) {
            source<<"real2 c0 = data"<<input<<"[base];\n";
            source<<"real2 c1 = data"<<input<<"[base+"<<(L*m)<<"];\n";
            source<<"data"<<output<<"[base+j*"<<m<<"] = c0+c1;\n";
            source<<"data"<<output<<"[base+(j+1)*"<<m<<"] = multiplyComplex(w[j*"<<zsize<<"/"<<(2*L)<<"], c0-c1);\n";
        }
        source<<"}\n";
        m = m*radix;
        source<<"__syncthreads();\n";
        source<<"}\n";
        ++stage;
    }

    // Create the kernel.

    bool outputIsReal = (inputIsReal && axis == 2 && !forward);
    bool outputIsPacked = (inputIsReal && axis == 2 && forward);
    string outputSuffix = (outputIsReal ? ".x" : "");
    if (outputIsPacked)
        source<<"if (index < XSIZE*YSIZE && x < XSIZE/2+1)\n";
    else
        source<<"if (index < XSIZE*YSIZE)\n";
    source<<"for (int i = threadIdx.x-block*THREADS_PER_BLOCK; i < ZSIZE; i += THREADS_PER_BLOCK)\n";
    if (outputIsPacked)
        source<<"out[y*(ZSIZE*(XSIZE/2+1))+i*(XSIZE/2+1)+x] = data"<<(stage%2)<<"[i+block*ZSIZE]"<<outputSuffix<<";\n";
    else
            source<<"out[y*(ZSIZE*XSIZE)+i*XSIZE+x] = data"<<(stage%2)<<"[i+block*ZSIZE]"<<outputSuffix<<";\n";
    map<string, string> replacements;
    replacements["XSIZE"] = context.intToString(xsize);
    replacements["YSIZE"] = context.intToString(ysize);
    replacements["ZSIZE"] = context.intToString(zsize);
    replacements["BLOCKS_PER_GROUP"] = context.intToString(blocksPerGroup);
    replacements["THREADS_PER_BLOCK"] = context.intToString(threadsPerBlock);
    replacements["M_PI"] = context.doubleToString(M_PI);
    replacements["COMPUTE_FFT"] = source.str();
    replacements["SIGN"] = (forward ? "1" : "-1");
    replacements["INPUT_TYPE"] = (inputIsReal && axis == 0 && forward ? "real" : "real2");
    replacements["OUTPUT_TYPE"] = (outputIsReal ? "real" : "real2");
    replacements["INPUT_IS_REAL"] = (inputIsReal && axis == 0 && forward ? "1" : "0");
    replacements["INPUT_IS_PACKED"] = (inputIsReal && axis == 0 && !forward ? "1" : "0");
    replacements["OUTPUT_IS_PACKED"] = (outputIsPacked ? "1" : "0");
    hipModule_t module = context.createModule(HipKernelSources::vectorOps+context.replaceStrings(HipKernelSources::fft, replacements));
    hipFunction_t kernel = context.getKernel(module, "execFFT");
    threads = blocksPerGroup*threadsPerBlock;
    return kernel;
}
