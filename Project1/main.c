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
    int threads = 1; 
    if (argc > 2) {
        printf("HERE\n");
        #ifdef PRODUCTION
        printf("Threads %d\n",atoi(argv[2])); 
        #endif
        threads = atoi(argv[2]);
    }
    int sum = 0;
    #ifdef ANNEALER
    int sum1 = 0;    
    rand_init(); 
    annealer_createInitialPlacement(layout);
    sum1 = netlist_layoutWirelength(layout);
    annealer_anneal(layout,sum1,threads);
    sum = netlist_layoutWirelength(layout);
    printf("Wirelength: %d %d\n",sum,sum1);
    #else 
    solver_quadraticWirelength(layout);
    #endif 
    sum = netlist_layoutWirelength(layout);
    printf("Wirelength: %d\n",sum);

    #ifdef PRODUCTION 
    netlist_printNetlist(layout); 
    netlist_printNetlist(layout); 
    netlist_printQoR(layout);
    //netlist_printForMatlab(layout);
    printf("done\n");
    #endif 
    #ifdef BENCHMARK 
    printf("Wirelength: %d\n",sum);
    #endif
    netlist_free(layout);
    return 0;
}

