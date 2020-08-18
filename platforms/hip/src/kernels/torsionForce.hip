const real PI = (real) 3.14159265358979323846;
real3 v0 = make_real3(pos1.x-pos2.x, pos1.y-pos2.y, pos1.z-pos2.z);
real3 v1 = make_real3(pos3.x-pos2.x, pos3.y-pos2.y, pos3.z-pos2.z);
real3 v2 = make_real3(pos3.x-pos4.x, pos3.y-pos4.y, pos3.z-pos4.z);
#if APPLY_PERIODIC
APPLY_PERIODIC_TO_DELTA(v0)
APPLY_PERIODIC_TO_DELTA(v1)
APPLY_PERIODIC_TO_DELTA(v2)
#endif
real3 cp0 = cross(v0, v1);
real3 cp1 = cross(v1, v2);
real cosangle = dot(normalize(cp0), normalize(cp1));
real theta;
if (cosangle > 0.99f || cosangle < -0.99f) {
    // We're close to the singularity in acos(), so take the cross product and use asin() instead.

    real3 cross_prod = cross(cp0, cp1);
    real scale = dot(cp0, cp0)*dot(cp1, cp1);
    theta = ASIN(SQRT(dot(cross_prod, cross_prod)/scale));
    if (cosangle < 0)
        theta = PI-theta;
}
else
   theta = ACOS(cosangle);
theta = (dot(v0, cp1) >= 0 ? theta : -theta);
COMPUTE_FORCE
real normCross1 = dot(cp0, cp0);
real normSqrBC = dot(v1, v1);
real normBC = SQRT(normSqrBC);
real normCross2 = dot(cp1, cp1);
real dp = RECIP(normSqrBC);
real4 ff = make_real4((-dEdAngle*normBC)/normCross1, dot(v0, v1)*dp, dot(v2, v1)*dp, (dEdAngle*normBC)/normCross2);
real3 force1 = ff.x*cp0;
real3 force4 = ff.w*cp1;
real3 s = ff.y*force1 - ff.z*force4;
real3 force2 = s-force1;
real3 force3 = -s-force4;
