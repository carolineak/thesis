// kernels.h
#ifndef KERNELS_H
#define KERNELS_H

// Returns 0 on success, -1 on failure.
// h_a, h_b, h_c are pointers to host arrays of length n.
int launch_vector_add(const float *h_a,
                      const float *h_b,
                      float *h_c,
                      int n);

#endif
