float4 exceptionParams = PARAMS[index];
real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#if APPLY_PERIODIC
APPLY_PERIODIC_TO_DELTA(delta)
#endif
real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
real invR = RSQRT(r2);
real sig2 = invR*exceptionParams.y;
sig2 *= sig2;
real sig6 = sig2*sig2*sig2;
real dEdR = exceptionParams.z*(12.0f*sig6-6.0f)*sig6;
real tempEnergy = exceptionParams.z*(sig6-1.0f)*sig6;
dEdR += exceptionParams.x*invR;
dEdR *= invR*invR;
tempEnergy += exceptionParams.x*invR;
energy += tempEnergy;
delta *= dEdR;
real3 force1 = -delta;
real3 force2 = delta;
