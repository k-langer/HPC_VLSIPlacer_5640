#define __jacobi__h
#ifdef __jacobi__h

extern float * jacobi_jacobi(float * A, float * b, int size, int itt);
extern void jacobi_jacobi2(float * resultx, float * resulty, float * A, float * bx, float * by, int size, int itt);
extern void jacobi_jacobi2s(float * resultx, float * resulty, float * A, float * bx, float * by, int size, int itt);
#endif
