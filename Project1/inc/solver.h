#define __solver__h
#ifdef __solver__h
/*
* The analytic quadratic wire length solver
* Algorithm ideas from coursera's class:
* "VLSI CAD: Logic to Layout" 
* Code written by Langer for HPC
*/
void solver_addGates(layout_t *, float **);
void solver_solve(layout_t *);
#endif 
