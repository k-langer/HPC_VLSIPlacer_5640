#include "netlist.h"
#include "parser.h"
#include "common.h" 
#include "annealer.h"
#include "rand.h"
#include "solver.h" 

int main(int argc, char**argv) {
    layout_t *layout = parser_parseNetlist(argv[1]); 
    if (!layout) {
        return -1;
    }
    /*    
    rand_init(); 
    annealer_createInitialPlacement(layout);
    int sum1 = netlist_layoutWirelength(layout);
    clock_t start = clock(), diff;
    annealer_anneal(layout,sum1);
    diff = clock() - start;
    int sum = netlist_layoutWirelength(layout);
    printf("Wirelength: %d %d\n",sum,sum1);
    */
    solver_quadraticWirelength(layout);
    
    //netlist_printNetlist(layout); 
    //netlist_printQoR(layout);
    /*
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Sequential time: %d s %d ms\n", msec/1000, msec%1000);
    netlist_printForMatlab(layout);
    netlist_free(layout);
    printf("done\n");
    */
    return 0;
}

