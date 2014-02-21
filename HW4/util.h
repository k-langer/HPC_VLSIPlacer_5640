#ifndef _UTIL_H_
#define _UTIL_H_
#include <stdlib.h>
#include <stdio.h>

/*
 * contains utility code that initializes and checks vectors.
 */

/*
 * parses and verifies cli input.
 * @argc - argument count
 * @argv - arguments
 
* @return - vector length
 */
size_t parse_args(int argc,char **argv){

  if(argc < 2){
    printf("need exactly 1 argument for vector length\n");
    return 0;
  }
  int vect_len = atoi(argv[1]);
  return vect_len;
}//parse_args

/*
 * generates the vectors. assumes that vectors have been allocated.
 * @vect_len - the length of the vectors
 * @vect1 - vector initialized to length vect_len
 * @vect2 - vector initialized to length vect_len
 */
void init_vects(size_t vect_len,float *vect1,float *vect2){
  for(size_t i = 0; i < vect_len; i++){
    vect1[i] = i;
    vect2[i] = i+i;
  }
}//init_vects

/*
 * @vect_len - the length of the vectors
 * @vect1 - vector initialized to length vect_len
 * @vect2 - vector initialized to length vect_len
 * @result - student generated addition of vect1 and vect2
 * @return - true if result is correct, otherwise false
 */
bool verify(size_t vect_len,float *vect1,float *vect2,float *result){
  float tmp_result = 0;
  for(size_t i = 0 ; i < vect_len; i++){
    tmp_result = vect1[i] + vect2[i];
    if(tmp_result != result[i]){
       return false;
    }
  }
  return true;
}//verify
#endif
