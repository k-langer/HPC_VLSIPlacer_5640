#define __solver__h
#ifdef __solver__h
/*
* The analytic quadratic wire length solver
* Algorithm ideas from coursera's class:
* "VLSI CAD: Logic to Layout" 
* Code written by Langer for HPC
*/
void solver_quadraticWirelength(layout_t *);
void solver_combinatoricsWire(wire_t * wire, float * C_matrix, int size);
void solver_fillCMatrix(layout_t * layout, float *  C_matrix,int size);
float * createMatrix(int ysize, int xsize);
void solver_printMatrix(float * C_matrix,int size_gates);
float * solver_fillAbMatrix(layout_t * layout, float * A_matrix, float * bx_matrix, float * by_matrix, float * C_matrix, int size_gates);
void solver_printAb(float * A, float * bx, float * by, int size);
float * solver_jacobi(float * A, float * b, int size, int itt);
void solver_assignGates(layout_t * layout, float * x, float * y, int size);
#endif 
