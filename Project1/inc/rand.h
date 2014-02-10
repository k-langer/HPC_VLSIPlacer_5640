/*
Uses Intel Ivy Bridge's entroy based random number generator.
Code heavily based on samples found here:
https://idea.popcount.org/2013-03-25-hardware-entropy-rdrand/
*/

#ifndef _RAND_H_
#define _RAND_H_

#include "common.h"

#define RDRAND_RETRY_LOOPS 10

#ifdef __x86_64__
# define RDRAND_LONG ".byte 0x48,0x0f,0xc7,0xf0"
#else
# define RDRAND_INT ".byte 0x0f,0xc7,0xf0"
# define RDRAND_LONG RDRAND_INT
#endif

int rand_init();
double rand_rdrand1();
int rand_rdrand(int *v, int range);

#endif
