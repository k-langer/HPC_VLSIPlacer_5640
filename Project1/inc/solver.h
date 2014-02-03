#ifdef __solver__h
#define __solver__h
/*
* The analytic quadratic wire length solver
* Algorithm ideas from coursera's class:
* "VLSI CAD: Logic to Layout" 
* Code written by Langer for HPC
*/
gate_t ** solver_solve(layout_t *);
#endif 
