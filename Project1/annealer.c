#include "annealer.h"
#include "rand.h"
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
wire_n * swap_gates(layout_t * layout, gate_n gaten, coord_t c2) {
    int count = 0; 
    gate_t * g1 = &(layout->all_gates[gaten]); 
    gate_t * g2 = layout->grid[c2.x+layout->x_size*c2.y];
    layout->grid[g1->x+layout->x_size*g1->y] = g2; 
    layout->grid[c2.x+layout->x_size*c2.y] = g1;
     
    if (g2) {
        g2->x = g1->x; 
        g2->y = g2->y;
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
    rand_init(); 
    if (!wirelength) {
        annealer_createInitialPlacement(layout); 
    }
    return layout;
}
