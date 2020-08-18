real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#if APPLY_PERIODIC
APPLY_PERIODIC_TO_DELTA(delta)
#endif
real r = SQRT(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
COMPUTE_FORCE
dEdR = (r > 0) ? (dEdR / r) : 0;
delta *= dEdR;
real3 force1 = delta;
real3 force2 = -delta;
