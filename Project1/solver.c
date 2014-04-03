#include "netlist.h"
#include "common.h"
//TODO: Implement numeric solver to get future GPU speedup 
/*
* For each wire, fill all relevant gates
*/ 
void solver_combinatoricsWire(wire_t * wire, float * C_matrix, int size) {
    int wc = wire->num_gates; 
    if( wc < 2 ) {
       return;  
    } 
    int x,y; 
    float weight = 1 / (wire->num_gates -1.0); 
    for (int i = 0; i < wc-1; i++) {
        x = wire->gates[i]; 
        for (int j = 1; j < wc; j++) {
            if (i != j) {
                y = wire->gates[j]; 
                //printf("%d %d %f\n",x,y,weight);
                C_matrix[y*size + x] += weight;
                C_matrix[x*size + y] += weight;  
            }
        }
    }
}
/*
* Create the connectivity matrix. 
*/
void solver_fillCMatrix(layout_t * layout, float *  C_matrix,int size) {
   for (int wireCount = 0; wireCount < layout->size_wires; wireCount++) {
       wire_t * wire = layout->all_wires + wireCount; 
       solver_combinatoricsWire(wire,C_matrix,size);
    } 
}
/* 
* return a row/col matrix full of all zeros
*/
float * createMatrix(int ysize, int xsize) {
    return calloc(sizeof(float),ysize*xsize); 
}
/*
* just print any square matrix to the console
*/
void solver_printMatrix(float * C_matrix,int size_gates) {
    for (int i = 0; i < size_gates; i++) {
        for (int j = 0; j < size_gates; j++) {
            printf("%f ",C_matrix[i*size_gates + j]);
        }
        printf("\n"); 
    }
}
/*
* Fill the A matrix (from the connectivity matrix) 
* also fill the bx and by matrix from the port locations
*/ 
float * solver_fillAbMatrix(layout_t * layout, float * A_matrix, float * bx_matrix, float * by_matrix, float * C_matrix, int size_gates) {
    float sum; 
    float elem; 
    for (int i = 0; i < size_gates; i++) {
        sum = 0; 
        for (int j = 0; j < size_gates; j++) {
            if (i != j) {
                elem = C_matrix[i*size_gates + j];
                sum += elem; 
                if (-1*elem < 0) {
                    A_matrix[i*size_gates + j] = -1*elem;
                }
            }
        }
        A_matrix[i*size_gates + i] = sum; 
    }
    port_t * port; 
    int x; 
    int y; 
    wire_t * wire; 
    gate_n gate;
    float weight;  
    for (int i = 0; i < layout->size_ports; i++) {
        port = layout->all_ports + i; 
        x = port->x; 
        y = port->y;
        wire = layout->all_wires + port->wire;
        weight = port->weight; 
        for (int j = 0; j < wire->num_gates; j++) {
            gate = wire->gates[j];
            A_matrix[gate*size_gates + gate] += weight; 
            bx_matrix[gate] += weight*x;
            by_matrix[gate] += weight*y;
        }
    }
    return A_matrix; 
}
/*
* This is the core of the solver using the Jacobi Method
* http://en.wikipedia.org/wiki/Jacobi_method
* Solve Ax = b
* where b is of size 'size' and A is a square matrix of size*size
* Becuase my connectivity matrix has values on every diagonal (becauase of the fact 
* that any gates IS connected to something) jacobi is garunteed to find some solution. 
* It is an itterative method, not an exact solution. It is MUCH faster than inverting the matrix and multiplying
* Code is based on algorithm in 'NUMERICAL METHODS for Mathematics, Science and Engineering, 2nd Ed, 1992' and the
* accompanying Matlab text.
*/
float * solver_jacobi(float * A, float * b, int size, int itt) {
    float Bv; 
    float Av; 
    float Xv;
    float * P = calloc(sizeof(float),size); 
    float * X = calloc(sizeof(float),size);  
    if (!P || !X) {
        printf("OUT OF MEMORY ERROR (jacobi)\n");
    } 
    for (int i = 0; i < size; i++) {
        P[i] = 0.0f;
        X[i] = 0.0f;
    }
    float v; 
    float * swap; 
    for (int i = 0; i < itt; i++) {
        for (int j = 0; j < size; j++) {
            Bv = b[j]; 
            Av = A[j*size+j]; 
            Xv = 0.0f; 
            for (int k = 0; k < size; k++) {
                if (j != k) {
                    v = A[j*size+k]*P[k];
                    if (v < 0 || v > 0) {
                        Xv+=v; 
                    }
                }
            }
            P[j] = (Bv - Xv)/Av; 
        }
        swap = P; 
        P = X; 
        X = swap; 
    }
    free(P); 
    return X; 
}
/*
* give a layout, preform a quadratic wirelength solve on it. 
* turns out it is way faster than simulated annealing
* Legalize cells, but do not provide any optimization on legalized cells 
*/
void solver_quadraticWirelength(layout_t * layout) {
    int size_gates = layout->size_gates; 
    float * C_matrix = createMatrix(size_gates,size_gates); 
    float * A_matrix = createMatrix(size_gates,size_gates); 
    float * bx_matrix = createMatrix(size_gates,1); 
    float * by_matrix = createMatrix(size_gates,1); 
    solver_fillCMatrix(layout,C_matrix,size_gates); 
    solver_fillAbMatrix(layout,A_matrix,bx_matrix,by_matrix, C_matrix,size_gates); 
    float * resultx = solver_jacobi(A_matrix,bx_matrix,size_gates,30);
    float * resulty = solver_jacobi(A_matrix,by_matrix,size_gates,30);
    //solver_printAb(A_matrix, bx_matrix, by_matrix, size_gates);
    //solver_printMatrix(A_matrix,size_gates);
    for (int i = 0; i < size_gates; i++) {
        printf("%f %f\n",resultx[i],resulty[i]);
    }
    printf("\n");
}
/*//DEPRECATED
//was being used to print matrix for use in MATLAB testing
//before I had implemented jacobi. Good stuff, 1+Gig text files. 
void solver_printAb(float * A, float * bx, float * by, int size) {
    FILE *fA = fopen("/media/kevin/UUI/matrix/A.txt", "w+");
    FILE *fbx = fopen("/media/kevin/UUI/matrix/bx.txt","w+");
    FILE *fby = fopen("/media/kevin/UUI/matrix/by.txt","w+"); 
    for (int i = 0; i < size*size; i++) {
        if (i %size == 0) {
            fprintf(fA,"\n");
        } 
        fprintf(fA,"%f ",A[i]);
        if (i < size) {
            fprintf(fbx,"%f\n",bx[i]);
            fprintf(fby,"%f\n",by[i]);
        }
    }
}
*/
