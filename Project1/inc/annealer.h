#ifndef __annealer__h
#define __annealer__h
#include "netlist.h"
#include "common.h"
layout_t * annealer_anneal(layout_t * layout, int wirelength);
gate_n * swap_gates(layout_t * , gate_n, coord_t);
#endif
