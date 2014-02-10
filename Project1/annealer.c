#include "annealer.h"
#include "rand.h"
#include "common.h" 
layout_t * annealer_createInitialPlacement(layout_t * layout) {
    int x_b = layout->x_size; 
    int y_b = layout->y_size; 
    int place_x, place_y; 
    for (int j = 0; j < layout->size_gates; j++) {
        do {
            rand_rdrand(&place_x, x_b); 
            rand_rdrand(&place_y, y_b);  
        } while(layout->grid[place_x+x_b*place_y]);
        layout->grid[place_x+x_b*place_y] = &(layout->all_gates[j]);  
        if (layout->all_gates[j].name) {
            layout->all_gates[j].x = place_x; 
            layout->all_gates[j].y = place_y; 
        }
    }
    return layout;  
}
wire_n * annealer_swapGates(layout_t * layout, gate_n gaten, coord_t c2) {
    int count = 0; 
    gate_t * g1 = &(layout->all_gates[gaten]); 
    gate_t * g2 = layout->grid[c2.x+layout->x_size*c2.y];
    layout->grid[g1->x+layout->x_size*g1->y] = g2; 
    layout->grid[c2.x+layout->x_size*c2.y] = g1;
     
    if (g2) {
        g2->x = g1->x; 
        g2->y = g1->y;
        count += 1 + g2->fanin_size; 
    }
    g1->x = c2.x;
    g1->y = c2.y;
    count += 1 + g1->fanin_size; 

    wire_n * recalc = malloc(sizeof(wire_n)*count+1); 
    count = 0; 
    recalc[0] = g1->fanout; 
    for (count = 1; count <= g1->fanin_size; count++) {
        recalc[count] = g1->fanin[count-1]; 
    }
    if (g2) {
        recalc[count] = g2->fanout; 
        count += 1; 
        for (int i = 0; i < g2->fanin_size; i++) {
            recalc[count+i] = g2->fanin[i]; 
        }
    }    
    return recalc; 
}
layout_t * annealer_anneal(layout_t * layout, int wirelength) {
    if (!wirelength) {
        annealer_createInitialPlacement(layout);
        wirelength = netlist_layoutWirelength(layout);  
    }
    return annealer_simulatedAnnealing(layout,wirelength,1000.0);
}
bool_t annealer_acceptSwap(int deltaL, double T) {
    return (exp(deltaL/T) > rand_rdrand1());
}
layout_t * annealer_simulatedAnnealing(
    layout_t * layout, int wirelength,double tempature) {
    int rand_gate; 
    coord_t swap_coor,swap_back;  
    wire_n * recalc; 
    int count; 
    int pre_wirelength, post_wirelength;
    int stall = 0; 
    int deltaT; 
    while (1) {
        pre_wirelength = 0;
        post_wirelength = 0;  
        rand_rdrand(&rand_gate, layout->size_gates); 
        rand_rdrand(&(swap_coor.x), layout->x_size);
        rand_rdrand(&(swap_coor.y), layout->y_size);
        swap_back.x = layout->all_gates[rand_gate].x;
        swap_back.y = layout->all_gates[rand_gate].y; 
        recalc = annealer_swapGates(layout,rand_gate,swap_coor);
        count = 0; 
        while (recalc[count]) {
            pre_wirelength += layout->all_wires[recalc[count]].wirelength;
            post_wirelength += netlist_wireWirelength(layout,recalc[count]);
            count++; 
        }
        free(recalc);
        deltaT = pre_wirelength - post_wirelength;  
        if (deltaT < 0 && !annealer_acceptSwap(deltaT,tempature)) {
            recalc = annealer_swapGates(layout,rand_gate,swap_back);
            count = 0; 
            while (recalc[count]) {
                netlist_wireWirelength(layout,recalc[count]);
                count++;
            }
            free(recalc);
        } else {
            if (deltaT == 0 && stall++ > 500) {
                break;
            } else if (deltaT > 0) {
                stall = 0;
                if (tempature > 0.01) {
                    tempature/=1.01;
                }
            }
            wirelength -= deltaT;  
            //printf("WL: %d T %f\n",wirelength,tempature);
        } 
    }   
    return layout;
}
