#include "netlist.h"
#include "common.h"
//TODO: Implement numeric solver to get future GPU speedup 

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
                C_matrix[y*size + x] = weight;
                C_matrix[x*size + y] = weight;  
            }
        }
    }
}
void solver_fillCMatrix(layout_t * layout, float *  C_matrix,int size) {
   for (int wireCount = 0; wireCount < layout->size_wires; wireCount++) {
       wire_t * wire = layout->all_wires + wireCount; 
       solver_combinatoricsWire(wire,C_matrix,size);
    } 
}
float * createMatrix(int ysize, int xsize) {
    return calloc(sizeof(float),ysize*xsize); 
}
void solver_printMatrix(float * C_matrix,int size_gates) {
    for (int i = 0; i < size_gates; i++) {
        for (int j = 0; j < size_gates; j++) {
            printf("%f ",C_matrix[i*size_gates + j]);
        }
        printf("\n"); 
    }
}
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
void solver_printAb(float * A, float * bx, float * by, int size) {
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
void solver_solve(layout_t * layout) {
    int size_gates = layout->size_gates; 
    float * C_matrix = createMatrix(size_gates,size_gates); 
    float * A_matrix = createMatrix(size_gates,size_gates); 
    float * bx_matrix = createMatrix(size_gates,1); 
    float * by_matrix = createMatrix(size_gates,1); 
    solver_fillCMatrix(layout,C_matrix,size_gates); 
    solver_fillAbMatrix(layout,A_matrix,bx_matrix,by_matrix, C_matrix,size_gates); 
    solver_printAb(A_matrix, bx_matrix, by_matrix, size_gates);
    //solver_printMatrix(A_matrix,size_gates);
}
