#include "netlist.h"
#include "common.h"
#include "sort.h" 
#include "jacobi.h"  
//TODO: Implement numeric solver to get future GPU speedup 
/*
* For each wire, fill all relevant gates
*/ 
void solver_combinatoricsWire(wire_t * wire, float * C_matrix, unsigned int size,unsigned int row) {
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
                C_matrix[y*row + x] += weight;
                C_matrix[x*row + y] += weight;  
            }
        }
    }
}
/*
* Create the connectivity matrix. 
*/
void solver_fillCMatrix(layout_t * layout, float *  C_matrix,unsigned int size, unsigned int row) {
   for (int wireCount = 0; wireCount < layout->size_wires; wireCount++) {
       wire_t * wire = layout->all_wires + wireCount; 
       solver_combinatoricsWire(wire,C_matrix,size,row);
    } 
}
/* 
* return a row/col matrix full of all zeros
*/
float * createMatrix(unsigned int ysize, unsigned int xsize) {
    float * ret; 
    #ifdef OPT
    ret = (float *) memalign(16,sizeof(float)*ysize*xsize);
    #endif
    ret = (float *) calloc(sizeof(float),ysize*xsize);
    if (!ret) {
        printf("OUT OF MEMORY ERROR\n");
    }
    return ret;
}
/*
* just print any square matrix to the console
*/
void solver_printMatrix(float * C_matrix,unsigned int size_gates) {
    for (unsigned int i = 0; i < size_gates; i++) {
        for (unsigned int j = 0; j < size_gates; j++) {
            printf("%f ",C_matrix[i*size_gates + j]);
        }
        printf("\n"); 
    }
}
/*
* Fill the A matrix (from the connectivity matrix) 
* also fill the bx and by matrix from the port locations
*/ 
float * solver_fillAMatrix(layout_t * layout, float * A_matrix, float * C_matrix, unsigned int size_gates,unsigned int row) {
    float sum; 
    float elem; 
    for (size_t i = 0; i < size_gates; i++) {
        sum = 0; 
        for (size_t j = 0; j < size_gates; j++) {
            if (i != j) {
                elem = C_matrix[i*row + j];
                sum += elem; 
                if (-1*elem < 0) {
                    A_matrix[i*row + j] = -1*elem;
                }
            }
        }
        if (sum < 0.000001) {
            sum = 0.000001; 
        }
        A_matrix[i*row + i] = sum; 
    }
    return A_matrix;
}
float * solver_fillAbMatrix(layout_t * layout, float * A_matrix, float * bx_matrix, float * by_matrix, unsigned int size_gates, unsigned int row) {
    port_t * port; 
    int x; 
    int y; 
    wire_t * wire; 
    gate_n gate;
    float weight;  
    for (size_t i = 0; i < layout->size_ports; i++) {
        port = layout->all_ports + i; 
        x = port->x; 
        y = port->y;
        wire = layout->all_wires + port->wire;
        weight = port->weight; 
        for (size_t j = 0; j < wire->num_gates; j++) {
            gate = wire->gates[j];
            A_matrix[gate*row + gate] += weight; 
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
float * solver_jacobi(float * restrict A, float * restrict b, unsigned int size, unsigned int row, unsigned int itt) {
    float * P = createMatrix(size,1);
    float * X = createMatrix(size,1);
    for (int i = 0; i < size; i++) {
        P[i] = 0.0f;
        X[i] = 0.0f;
    }
    float * swap; 
    size_t i, j;
    for (i = 0; i < itt; i++) {
        #pragma omp parallel for private(j)
        for (j = 0; j < size; j++) {
            float Bv, Av, Xv; 
            Bv = b[j]; 
            Av = A[j*row+j]; 
            Xv = 0.0f;
            #ifdef OPT
            float *P = __builtin_assume_aligned(P, 32);
            float *A = __builtin_assume_aligned(A, 32);
            for (int k = 0; k < size; k+=8) {
                Xv += A[j*size+k]*P[k];
                Xv += A[j*size+k+1]*P[k+1];
                Xv += A[j*size+k+2]*P[k+2];
                Xv += A[j*size+k+3]*P[k+3];
                Xv += A[j*size+k+4]*P[k+4];
                Xv += A[j*size+k+5]*P[k+5];
                Xv += A[j*size+k+6]*P[k+6];
                Xv += A[j*size+k+7]*P[k+7];
            }
            #else
            for (int k = 0; k < size; k++) {
                Xv += A[j*size+k]*P[k];
            }
            #endif
            Xv -= Av*P[j];
            P[j] = (Bv - Xv)/Av; 
        }
        swap = P; 
        P = X; 
        X = swap; 
    }
    free(P); 
    return X; 
}
int solver_assignGate(layout_t * layout, int * xr, int * yr) {
    int x = *xr; 
    int y = *yr; 
    int xsize = layout->x_size; 
    int ysize = layout->y_size; 
    gate_t ** grid = layout->grid;
    if (grid[y*xsize + x] == 0) {
        return 0; 
    }
    int scale = 0; 
    while (1) {
        scale += 1;
        for (int j = y-scale; j < y+scale; j++) {
            for (int i = x-scale; i < x+scale; i++) {
                //printf("x %d y %d xs %d xy %d\n",x,y,i,j); 
                if (j >= 0 && j < ysize && i >= 0 && i < xsize && grid[j*xsize + i] == 0) {
                    *xr = i;
                    *yr = j;
                    return 0;
                }
            }
        }
        if (scale > xsize || scale > ysize) {
            break; 
        }
    }
    printf("Could not do\n");
    return -1;         
}
void solver_assignGates(layout_t * layout, float * x, float * y, unsigned int size) {
    printf("Assinging gates\n");
    int xr, yr;
    //char * visited = calloc(sizeof(char),layout->x_size*layout->y_size);
    for (int i = 0; i < size; i++) {
        gate_t * gate = layout->all_gates+i; 
        //solver_assignGate(layout,visited,i,0,INT_MAX,rint( x[i] ),rint( y[i] ),&xr,&yr);
        xr = rint( x[i] ); 
        yr = rint( y[i] );
        solver_assignGate(layout, &xr, &yr);
        layout->grid[yr*layout->x_size + xr] = gate;
        gate->x = xr;
        gate->y = yr;
    }
}
void solver_clearPlacement(layout_t * layout) {
}
//was being used to print matrix for use in MATLAB testing
//before I had implemented jacobi. Good stuff, 1+Gig text files. 
void solver_printAb(float * A, float * bx, float * by, unsigned int size) {
    FILE *fA = fopen("matrix/A.txt", "w+");
    FILE *fbx = fopen("matrix/bx.txt","w+");
    FILE *fby = fopen("matrix/by.txt","w+"); 
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
/*
* give a layout, preform a quadratic wirelength solve on it. 
* turns out it is way faster than simulated annealing
* Legalize cells, but do not provide any optimization on legalized cells 
*/
void solver_quadraticWirelength(layout_t * layout) {
    int size_gates = layout->size_gates;
    int row_size = size_gates; 
    #ifdef OPT
    int row_size = (size_gates+7) & ~7UL; 
    #endif
    float * C_matrix = createMatrix(row_size,row_size); 
    float * A_matrix = createMatrix(row_size,row_size); 
    float * bx_matrix = createMatrix(row_size,1); 
    float * by_matrix = createMatrix(row_size,1); 
    solver_fillCMatrix(layout,C_matrix,size_gates,row_size); 
    solver_fillAMatrix(layout,A_matrix,C_matrix,size_gates,row_size);
    solver_fillAbMatrix(layout,A_matrix,bx_matrix,by_matrix,size_gates,row_size); 
    #ifndef CUDA
    float * resultx = solver_jacobi(A_matrix,bx_matrix,size_gates,row_size,30);
    float * resulty = solver_jacobi(A_matrix,by_matrix,size_gates,row_size,30);
    #else 
    float * resultx =  createMatrix(1,size_gates);
    float * resulty =  createMatrix(1,size_gates);
    jacobi_jacobi2(resultx,resulty,A_matrix,bx_matrix,by_matrix,size_gates,30);
    //float * resultx = jacobi_jacobi(A_matrix,bx_matrix,size_gates,30);
    //float * resulty = jacobi_jacobi(A_matrix,by_matrix,size_gates,30);
    #endif
    //solver_printAb(A_matrix, bx_matrix, by_matrix, size_gates);
    //solver_printMatrix(A_matrix,size_gates);
    int xr,yr;
    for (int i = 0; i < size_gates; i++ ) {
        gate_t * gate = layout->all_gates+i; 
        xr = rint( resultx[i] ); 
        yr = rint( resulty[i] );
        gate->x = xr;
        gate->y = yr;
    }
//    solver_assignGates(layout,resultx,resulty,size_gates);
}


/*
// Attempt at a roll-my-own A* search without a specfic goal
// The idea is to search through all surrounding nodes and keep a
// visited cache. Accepting the first empty space. 
// Basically a complete failure
void solver_assignGate(layout_t * layout, char * visited, int gate, int depth, int ldepth, int x, int y, int *xr, int *yr) {
    int place = y*layout->x_size + x; 
    if ( ldepth > depth 
        && (x >= 0 && x < layout->x_size) 
        && (y >= 0 && y < layout->y_size) 
        && visited[place] != gate+1) {
            visited[place] = gate+1; 
            if (!layout->grid[place]) {
                *yr = y; 
                *xr = x; 
                ldepth = depth; 
            } 
            solver_assignGate(layout,visited, gate, depth+1, ldepth, x+1,y,xr,yr);
            solver_assignGate(layout,visited, gate, depth+1, ldepth, x,y+1,xr,yr);
            solver_assignGate(layout,visited, gate, depth+1, ldepth, x-1,y,xr,yr);
            //solver_assignGate(layout,visited, gate, depth+1, ldepth, x,y-1,xr,yr); 
    }
}
*/
