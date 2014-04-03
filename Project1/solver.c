#include "netlist.h"
#include "common.h"
//TODO: Implement numeric solver to get future GPU speedup 

void solver_combinatoricsWire(wire_t * wire, float ** C_matrix) {
    int wc = wire->num_gates; 
    int x,y; 
    if( wc < 2 ) {
       return;  
    } 
    float weight = 1 / (wc + 0.0); 
    printf("%s %d\n",wire->name,wc); 
    for (int i = 0; i < wc-1; i++) {
        for (int j = 1; j < wc; j++) {
            if (i != j) {
                x = wire->gates[i]; 
                y = wire->gates[j]; 
                printf("\t(%d,%d)\n",x,y); 
                //C_matrix[0][0] = weight;
                //C_matrix[0][0] = weight;  
            }
        }
    }
}
void solver_addGates(layout_t * layout, float **  C_matrix) {
   for (int wireCount = 0; wireCount < layout->size_wires; wireCount++) {
       wire_t * wire = layout->all_wires + wireCount; 
       solver_combinatoricsWire(wire,C_matrix);
    } 
}
void solver_solve(layout_t * layout) {
    int size_gates = layout->size_gates; 
    float C_matrix[size_gates][size_gates];
    solver_addGates(layout,(float **)C_matrix); 
    printf("%d \n",layout->x_size); 
}
