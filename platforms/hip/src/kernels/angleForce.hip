real3 v0 = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
real3 v1 = make_real3(pos2.x-pos3.x, pos2.y-pos3.y, pos2.z-pos3.z);
#if APPLY_PERIODIC
APPLY_PERIODIC_TO_DELTA(v0)
APPLY_PERIODIC_TO_DELTA(v1)
#endif
real3 cp = cross(v0, v1);
real rp = cp.x*cp.x + cp.y*cp.y + cp.z*cp.z;
rp = max(SQRT(rp), (real) 1.0e-06f);
real r21 = v0.x*v0.x + v0.y*v0.y + v0.z*v0.z;
real r23 = v1.x*v1.x + v1.y*v1.y + v1.z*v1.z;
real dot = v0.x*v1.x + v0.y*v1.y + v0.z*v1.z;
real cosine = min(max(dot*RSQRT(r21*r23), (real) -1), (real) 1);
real theta = ACOS(cosine);
COMPUTE_FORCE
real3 force1 = cross(v0, cp)*(dEdAngle/(r21*rp));
real3 force3 = cross(cp, v1)*(dEdAngle/(r23*rp));
real3 force2 = -force1-force3;
