#include "sort.h"
/* Quicksort 
* from http://www.comp.dit.ie/rlawlor/Alg_DS/sorting/quickSort.c
* Accessed on April 8th 2014
* modified to accept two generic arrays
*/


void quickSort( float *a, int * b, int l, int r)
{
   int j;

   if( l < r ) 
   {
    // divide and conquer
        j = partition( a, b, l, r);
       quickSort( a, b, l, j-1);
       quickSort( a, b, j+1, r);
   }
    
}

int partition( float *a, int *b, int l, int r) {
   int pivot, i, j, t1,t2;
   pivot = a[l];
   i = l; j = r+1;
        
   while( 1)
   {
    do ++i; while( a[i] <= pivot && i <= r );
    do --j; while( a[j] > pivot );
    if( i >= j ) break;
    t1 = a[i]; a[i] = a[j]; a[j] = t1;
    t2 = b[i]; b[i] = b[j]; b[j] = t2;
   }
   t1 = a[l]; a[l] = a[j]; a[j] = t1;
   t2 = b[l]; b[l] = b[j]; b[j] = t2;
   return j;
}
