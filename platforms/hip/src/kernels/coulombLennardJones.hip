{
#if USE_EWALD
    unsigned int includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
    const real alphaR = EWALD_ALPHA*r;
    const real expAlphaRSqr = EXP(-alphaR*alphaR);
#if HAS_COULOMB
    const real prefactor = 138.935456f*CHARGE1*CHARGE2*invR;
#else
    const real prefactor = 0.0f;
#endif

#ifdef USE_DOUBLE_PRECISION
    const real erfcAlphaR = erfc(alphaR);
#else
    // This approximation for erfc is from Abramowitz and Stegun (1964) p. 299.  They cite the following as
    // the original source: C. Hastings, Jr., Approximations for Digital Computers (1955).  It has a maximum
    // error of 1.5e-7.

    const real t = RECIP(1.0f+0.3275911f*alphaR);
    const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
    real tempForce = 0.0f;
#if HAS_LENNARD_JONES
    real sig = SIGMA_EPSILON1.x + SIGMA_EPSILON2.x;
    real sig2 = invR*sig;
    sig2 *= sig2;
    real sig6 = sig2*sig2*sig2;
    real eps = SIGMA_EPSILON1.y*SIGMA_EPSILON2.y;
    real epssig6 = sig6*eps;
    tempForce = epssig6*(12.0f*sig6 - 6.0f);
    real ljEnergy = epssig6*(sig6 - 1.0f);
    #if USE_LJ_SWITCH
    if (r > LJ_SWITCH_CUTOFF) {
        real x = r-LJ_SWITCH_CUTOFF;
        real switchValue = 1+x*x*x*(LJ_SWITCH_C3+x*(LJ_SWITCH_C4+x*LJ_SWITCH_C5));
        real switchDeriv = x*x*(3*LJ_SWITCH_C3+x*(4*LJ_SWITCH_C4+x*5*LJ_SWITCH_C5));
        tempForce = tempForce*switchValue - ljEnergy*switchDeriv*r;
        ljEnergy *= switchValue;
    }
    #endif
#if DO_LJPME
    // The multiplicative term to correct for the multiplicative terms that are always
    // present in reciprocal space.
    const real dispersionAlphaR = EWALD_DISPERSION_ALPHA*r;
    const real dar2 = dispersionAlphaR*dispersionAlphaR;
    const real dar4 = dar2*dar2;
    const real dar6 = dar4*dar2;
    const real invR2 = invR*invR;
    const real expDar2 = EXP(-dar2);
    const float2 sigExpProd = SIGMA_EPSILON1*SIGMA_EPSILON2;
    const real c6 = 64*sigExpProd.x*sigExpProd.x*sigExpProd.x*sigExpProd.y;
    const real coef = invR2*invR2*invR2*c6;
    const real eprefac = 1.0f + dar2 + 0.5f*dar4;
    const real dprefac = eprefac + dar6/6.0f;
    // The multiplicative grid term
    ljEnergy += coef*(1.0f - expDar2*eprefac);
    tempForce += 6.0f*coef*(1.0f - expDar2*dprefac);
    // The potential shift accounts for the step at the cutoff introduced by the
    // transition from additive to multiplicative combintion rules and is only
    // needed for the real (not excluded) terms.  By addin these terms to ljEnergy
    // instead of tempEnergy here, the includeInteraction mask is correctly applied.
    sig2 = sig*sig;
    sig6 = sig2*sig2*sig2*INVCUT6;
    epssig6 = eps*sig6;
    // The additive part of the potential shift
    ljEnergy += epssig6*(1.0f - sig6);
    // The multiplicative part of the potential shift
    ljEnergy += MULTSHIFT6*c6;
#endif
    tempForce += prefactor*(erfcAlphaR+alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
    tempEnergy += includeInteraction ? ljEnergy + prefactor*erfcAlphaR : 0;
#else
    tempForce = prefactor*(erfcAlphaR+alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
    tempEnergy += includeInteraction ? prefactor*erfcAlphaR : 0;
#endif
    dEdR += includeInteraction ? tempForce*invR*invR : 0;
#else
#ifdef USE_CUTOFF
    unsigned int includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
#else
    unsigned int includeInteraction = (!isExcluded);
#endif
    real tempForce = 0.0f;
#if HAS_LENNARD_JONES
    real sig = SIGMA_EPSILON1.x + SIGMA_EPSILON2.x;
    real sig2 = invR*sig;
    sig2 *= sig2;
    real sig6 = sig2*sig2*sig2;
    real epssig6 = sig6*(SIGMA_EPSILON1.y*SIGMA_EPSILON2.y);
    tempForce = epssig6*(12.0f*sig6 - 6.0f);
    real ljEnergy = includeInteraction ? epssig6*(sig6 - 1) : 0;
    #if USE_LJ_SWITCH
    if (r > LJ_SWITCH_CUTOFF) {
        real x = r-LJ_SWITCH_CUTOFF;
        real switchValue = 1+x*x*x*(LJ_SWITCH_C3+x*(LJ_SWITCH_C4+x*LJ_SWITCH_C5));
        real switchDeriv = x*x*(3*LJ_SWITCH_C3+x*(4*LJ_SWITCH_C4+x*5*LJ_SWITCH_C5));
        tempForce = tempForce*switchValue - ljEnergy*switchDeriv*r;
        ljEnergy *= switchValue;
    }
    #endif
    tempEnergy += ljEnergy;
#endif
#if HAS_COULOMB
  #ifdef USE_CUTOFF
    const real prefactor = 138.935456f*CHARGE1*CHARGE2;
    tempForce += prefactor*(invR - 2.0f*REACTION_FIELD_K*r2);
    tempEnergy += includeInteraction ? prefactor*(invR + REACTION_FIELD_K*r2 - REACTION_FIELD_C) : 0;
  #else
    const real prefactor = 138.935456f*CHARGE1*CHARGE2*invR;
    tempForce += prefactor;
    tempEnergy += includeInteraction ? prefactor : 0;
  #endif
#endif
    dEdR += includeInteraction ? tempForce*invR*invR : 0;
#endif
}
