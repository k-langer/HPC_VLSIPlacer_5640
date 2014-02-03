/*
Uses Intel Ivy Bridge's entroy based random number generator.
Code heavily based on samples found here:
https://idea.popcount.org/2013-03-25-hardware-entropy-rdrand/
*/

#include "rand.h"

int rand_rdrand(int *v, int range) {
   int ok;
   asm volatile("1: " RDRAND_LONG "\n\t"
             "jc 2f\n\t"
             "decl %0\n\t"
             "jnz 1b\n\t"
             "2:"
             : "=r" (ok), "=a" (*v)
             : "0" (RDRAND_RETRY_LOOPS));
   *v = (((unsigned int)*v>>16)*range)>>16;
   return ok;
}


