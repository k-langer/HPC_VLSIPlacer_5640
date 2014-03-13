#include "netlist.h"
#include "parser.h"
#include "common.h" 
#include "solver.h" 
#include "annealer.h"
#include "rand.h"

void production(layout_t * layout, int threads);
void benchmark(layout_t * layout, char * name);
int main(int argc, char**argv) {
    layout_t *layout = parser_parseNetlist(argv[1]); 
    if (!layout) {
        return -1;
    }
    rand_init(); 
    #ifndef PRODUCTION
    benchmark(layout,argv[1]); 
    #endif
    #ifdef PRODUCTION
    production(layout,8); 
    #endif
    return 0;
}
void production(layout_t * layout, int threads) {
    struct timeval start, end;
    annealer_createInitialPlacement(layout);
    int sum1 = netlist_layoutWirelength(layout);
    gettimeofday(&start, NULL);
    annealer_anneal(layout,sum1,threads);
    gettimeofday(&end, NULL);
    int sum = netlist_layoutWirelength(layout);
    printf("Wirelength: %d %d\n",sum,sum1);
    netlist_printNetlist(layout); 
    netlist_printQoR(layout);
    netlist_verifyResults(layout);
    netlist_free(layout);
    long int sec = ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec));
    printf("Time: %lu s %lu ms\n", sec/1000000, sec%1000000);

}
void benchmark(layout_t * layout, char * name) {
    struct timeval start, end;
    FILE *fp = fopen("bench_results.txt", "w+");
    fprintf(fp,"# Benchmark results (5)(16)\n");
    for (int iterations = 0; iterations < 3; iterations++) {
        for (int threads = 1; threads <= 4; threads++) {
            layout_t *layout = parser_parseNetlist(name); 
            annealer_createInitialPlacement(layout);
            int sum1 = netlist_layoutWirelength(layout);
            gettimeofday(&start, NULL);
            annealer_anneal(layout,sum1,threads);
            gettimeofday(&end, NULL);
            int sum = netlist_layoutWirelength(layout);
            long int sec = ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec));
            netlist_free(layout);
            fprintf(fp,"%d\t%ld\t%ld\t%d\t%d\n",threads,sec/1000000,sec,sum,sum1);
            //netlist_free(layout);
        }
    }
}

