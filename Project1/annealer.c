#include "annealer.h"
#include "rand.h"
#include "common.h"
/*All code my own, but ideas from UIUC /VLSI CAD: Logic to Layout/
 *its a great Coursera course. highly recommend it 
*/

/*The annealer, as written must start with a placement. 
 *As a result, this function creates a random placement. 
 *This placement is terrible, and will be improved upon
*/
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
/*
* Swap a gate in the layout (specified with gate pointer gaten) with 
* any coord on the grid in layout. Note, everything must snap to grid
*/
wire_n * annealer_swapGates(layout_t * layout, gate_n gaten, coord_t c2, int * count_ptr) {
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
    *count_ptr += count; 
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
    *count_ptr += count; 
    return recalc; 
}
/*
* Wrapper function for the annealing. 
* -Creates placement if needed
* -Sets the initial tempature for annealing
*/
layout_t * annealer_anneal(layout_t * layout, int wirelength) {
    if (!wirelength) {
        annealer_createInitialPlacement(layout);
        wirelength = netlist_layoutWirelength(layout);  
    }
    return annealer_simulatedAnnealing(layout,wirelength,200.0);
}
/*
 * Use probablity magic to determine if a adverse swap is accepted
 * T is assumed to be > 0 and deltaL is assumed to be >= 0
*/
bool_t annealer_acceptSwap(int deltaL, double T) {
    #ifdef NO_ANNEALER
        return FALSE;
    #endif
    if (T > 0) {
        return (exp(deltaL/T) > rand_rdrand1());
    } else {
        return FALSE;
    }
}
/* Do the annealing. Swap randomly. Accept all swaps that impove
 * placement. Accept negative swaps that pass the acceptSwap test
 * based on their probalities and effect. Quit out of the loop
 * after 500 consequtive stalls defined as swaps that do not change
 * state of netlist. 
 * TODO: Better pick tempature value and stall value for max QoR
*/
layout_t * annealer_simulatedAnnealing(
    layout_t * layout, int wirelength,double tempature) {
    int rand_gate; 
    coord_t swap_coor,swap_back;  
    wire_n * recalc; 
    int count; 
    int pre_wirelength, post_wirelength;
    int stall = 0; 
    int deltaT; 
    int printWL = 100000000;
    while (1) {
        pre_wirelength = 0;
        post_wirelength = 0;  
        rand_rdrand(&rand_gate, layout->size_gates); 
        rand_rdrand(&(swap_coor.x), layout->x_size);
        rand_rdrand(&(swap_coor.y), layout->y_size);
        swap_back.x = layout->all_gates[rand_gate].x;
        swap_back.y = layout->all_gates[rand_gate].y; 
        //Make a random swap
        count = 0;
        recalc = annealer_swapGates(layout,rand_gate,swap_coor,&count);
        for (int i = 0; i < count; i++) {
            pre_wirelength += layout->all_wires[recalc[i]].wirelength;
            post_wirelength += netlist_wireWirelength(layout,recalc[i]);
        }    
        free(recalc);
        deltaT = pre_wirelength - post_wirelength;  
        if ((deltaT < 0 ) && !annealer_acceptSwap(deltaT,tempature)) {
            //Swap back rejects
            //TODO: optimize this 
            recalc = annealer_swapGates(layout,rand_gate,swap_back, &count);
            count = 0; 
            while (recalc[count]) {
                netlist_wireRevertWirelength(layout,recalc[count]);
                count++;
            }
            free(recalc);
        } else {
            if (deltaT <= 0 && stall++ > 500) {
                break;
            } else if (deltaT > 0) {
                stall = 0;
                //tempature /=1.00001;
                tempature-=0.0001;
            }
            wirelength -= deltaT; 
            //Display after 10k changes
            if (printWL > wirelength + 10000) {
                printWL = wirelength;  
                printf("WL: %d T %f\n",printWL,tempature);
            }
        } 
    }   
    return layout;
}
