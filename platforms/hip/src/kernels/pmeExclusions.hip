const float4 exclusionParams = PARAMS[index];
real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
const real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
const real r = SQRT(r2);
const real invR = RECIP(r);
const real alphaR = EWALD_ALPHA*r;
const real expAlphaRSqr = EXP(-alphaR*alphaR);
real tempForce = 0.0f;
if (alphaR > 1e-6f) {
    const real erfAlphaR = ERF(alphaR);
    const real prefactor = exclusionParams.x*invR;
    tempForce = -prefactor*(erfAlphaR-alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
    energy -= prefactor*erfAlphaR;
}
else {
    energy -= TWO_OVER_SQRT_PI*EWALD_ALPHA*exclusionParams.x;
}
#if DO_LJPME
const real dispersionAlphaR = EWALD_DISPERSION_ALPHA*r;
const real dar2 = dispersionAlphaR*dispersionAlphaR;
const real dar4 = dar2*dar2;
const real dar6 = dar4*dar2;
const real invR2 = invR*invR;
const real expDar2 = EXP(-dar2);
const real c6 = 64*exclusionParams.y*exclusionParams.y*exclusionParams.y*exclusionParams.z;
const real coef = invR2*invR2*invR2*c6;
const real eprefac = 1.0f + dar2 + 0.5f*dar4;
const real dprefac = eprefac + dar6/6.0f;
energy += coef*(1.0f - expDar2*eprefac);
tempForce += 6.0f*coef*(1.0f - expDar2*dprefac);
#endif
if (r > 0)
    delta *= tempForce*invR*invR;
real3 force1 = -delta;
real3 force2 = delta;
