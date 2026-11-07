#ifndef Q6K_XM_PROFILE_H
#define Q6K_XM_PROFILE_H
#include <stdint.h>
/* Summed worker durations (not wall time) when workers > 1. */
struct Q6kXmProfile { uint64_t activation, weights, matrix, reduce, calls, workers; };
extern _Thread_local struct Q6kXmProfile q6k_xm_profile;
#endif
