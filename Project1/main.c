#include "netlist.h"
#include "parser.h"
#include "common.h" 
#include "solver.h" 
#include "annealer.h"
#include "rand.h"
int main() {
    layout_t *layout = parser_parseNetlist("netlist.txt"); 
    if (!layout) {
        return -1;
    }
    rand_init(); 
    annealer_createInitialPlacement(layout);
    int sum1 = netlist_layoutWirelength(layout);
    annealer_anneal(layout,sum1);
    int sum = netlist_layoutWirelength(layout);
    printf("Wirelength: %d %d\n",sum,sum1);
    netlist_printNetlist(layout); 
    return 0;
}
