#include "netlist.h"
#include "parser.h"
#include "common.h" 
#include "solver.h" 
#include "annealer.h"
#include "rand.h"
int main(int argc, char**argv) {
    layout_t *layout = parser_parseNetlist(argv[1]); 
    if (!layout) {
        return -1;
    }
    rand_init(); 
    annealer_createInitialPlacement(layout);
    int sum1 = netlist_layoutWirelength(layout);
    clock_t start = clock(), diff;
    annealer_anneal(layout,sum1);
    diff = clock() - start;
    int sum = netlist_layoutWirelength(layout);
    printf("Wirelength: %d %d\n",sum,sum1);
    netlist_printNetlist(layout); 
    netlist_printQoR(layout);
    netlist_free(layout);
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Sequential time: %d s %d ms\n", msec/1000, msec%1000);
    return 0;
}

