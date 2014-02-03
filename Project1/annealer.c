#include "annealer.h"
#include "rand.h"
layout_t * annealer_anneal(layout_t * layout, int wirelength) {
    if (wirelength) {
    } else {
        srand(time(NULL));
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
    }
    return layout;
}
