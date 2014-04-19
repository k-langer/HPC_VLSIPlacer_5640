/*
Uses Intel Ivy Bridge's entroy based random number generator.
Code heavily based on samples found here:
https://idea.popcount.org/2013-03-25-hardware-entropy-rdrand/
(and above website borrows from the linux kernel)
*/

#include "rand.h"
int rand_init() {
    #ifndef IVYHASWELL
    /*If not haswell or ivy bridge just use rand()
    * I did breif timing and QoR studies and it seems 
    * like they provide similar results
    */
    #ifdef PRODUCTION
    srand(time(NULL));
    #else 
    srand(777);
    #endif
    #endif
    return TRUE; 
}
double rand_rdrand1() {
   #ifdef IVYHASWELL
   int n;
   int * v = &n;
   int ok; 
   asm volatile("1: " RDRAND_LONG "\n\t"
             "jc 2f\n\t"
             "decl %0\n\t"
             "jnz 1b\n\t"
             "2:"
             : "=r" (ok), "=a" (*v)
             : "0" (RDRAND_RETRY_LOOPS));
   *v = (((unsigned int)*v>>16));
   return (*v)/(65536.0);
   #endif
   return ((double) rand() / (RAND_MAX));
}
int rand_rdrand(int *v, int range) {
   #ifdef IVYHASWELL
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
   #endif
   *v = rand()%range;  
   return TRUE;
}


