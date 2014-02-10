#ifndef __annealer__h
#define __annealer__h
#include "netlist.h"
#include "common.h"
layout_t * annealer_createInitialPlacement(layout_t * );
layout_t * annealer_anneal(layout_t * layout, int wirelength);
gate_n * annealer_swapGates(layout_t * , gate_n, coord_t);
layout_t * annealer_simulatedAnnealing(layout_t *, int, double);
bool_t annealer_acceptSwap(int , double);
#endif
