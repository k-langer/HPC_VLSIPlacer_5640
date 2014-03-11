#include "netlist.h"
#include "parser.h"
#include "common.h" 
#include "solver.h" 
#include "annealer.h"
#include "rand.h"

void production(layout_t * layout);
void benchmark(layout_t * layout, char * name);
int main(int argc, char**argv) {
    layout_t *layout = parser_parseNetlist(argv[1]); 
    if (!layout) {
        return -1;
    }
    rand_init(); 
    benchmark(layout,argv[1]);
    //production(layout); 
    return 0;
}
void production(layout_t * layout) {
    annealer_createInitialPlacement(layout);
    int sum1 = netlist_layoutWirelength(layout);
    clock_t start = clock(), diff;
    annealer_anneal(layout,sum1,4);
    diff = clock() - start;
    int sum = netlist_layoutWirelength(layout);
    printf("Wirelength: %d %d\n",sum,sum1);
    netlist_printNetlist(layout); 
    netlist_printQoR(layout);
    netlist_verifyResults(layout);
    netlist_free(layout);
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time: %d s %d ms\n", msec/1000, msec%1000);

}
void benchmark(layout_t * layout, char * name) {
    struct timeval start, end;
    for (int iterations = 0; iterations < 5; iterations++) {
        for (int threads = 1; threads < 16; threads++) {
            layout_t *layout = parser_parseNetlist(name); 
            annealer_createInitialPlacement(layout);
            int sum1 = netlist_layoutWirelength(layout);
            gettimeofday(&start, NULL);
            annealer_anneal(layout,sum1,threads);
            gettimeofday(&end, NULL);
            int sum = netlist_layoutWirelength(layout);
            long int sec = ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec));
            netlist_free(layout);
            printf("%d\t%ld\t%ld\t%d\t%d\n",threads,sec/1000000,sec,sum,sum1);
            //netlist_free(layout);
        }
    }
}

