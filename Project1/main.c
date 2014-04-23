#include "netlist.h"
#include "parser.h"
#include "common.h" 
#include "annealer.h"
#include "rand.h"
#include "solver.h" 

int main(int argc, char**argv) {
    layout_t *layout = parser_parseNetlist(argv[1]); 
    struct timeval start, end;
    if (!layout) {
        return -1;
    }
    int threads = 1; 
    if (argc > 2) {
        #ifdef PRODUCTION
        printf("Threads %d\n",atoi(argv[2])); 
        #else 
        printf("THDS: %d",atoi(argv[2])); 
        #endif
        threads = atoi(argv[2]);
    }
    gettimeofday(&start,NULL);
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
    omp_set_dynamic(0); 
    omp_set_num_threads(threads);
    solver_quadraticWirelength(layout);
    #endif 
    
    gettimeofday(&end,NULL); 
    long int sec = ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec*1000000+start.tv_usec)); 
    sum = netlist_layoutWirelength(layout);
    

    #ifdef PRODUCTION 
    netlist_printNetlist(layout); 
    netlist_printNetlist(layout); 
    netlist_printQoR(layout);
    //netlist_printForMatlab(layout);
    printf("Wirelength: %d\n Seconds: %lu\n",sum,sec/1000000);
    printf("done\n");
    #else
    printf(" WL: %d TIME: %lu.%lu\n",sum,sec/1000000,(sec/10000)%100);
    #endif
    netlist_free(layout);
    return 0;
}

