#ifndef __netlist__h
#define __netlist__h
typedef int wire_n; 
typedef int gate_n;
typedef int port_n; 
/* Super Special Note, for all structures the largest bitwidth items
* are placed at the top of the structure. This was to maximize packing
* and minimize D$ problems. So that is why everything is in wony orders
*/

typedef struct hier {
    char * label;     
    struct hier * next; 
    double density; 
    int x; 
    int y; 
    unsigned int size;
} hier_t; 
/*
* Port:
*  -Name: of power 
*  -X: placed location
*  -y: placed location
*  -wire:wire connected to port
*/
typedef struct port { 
    char * name; 
    wire_n wire; 
    int x; 
    int y;  
} port_t; 
/*
* Gate:
*   -Name: of gate
*   -fanin: array of wire pointers, fanin
*   -fanout: wire attached to output
*   -x: placed location
*   -y: placed location
*/
typedef struct gate {
    char * name; 
    wire_n * fanin; 
    wire_n fanout; 
    int fanin_size; 
    int x; 
    int y;
} gate_t; 
/*
* Wire:
*   name: of port
*   gates: attached to wire
*   ports: attached to wire
*   wirelength: bounding box wirelength
*/
typedef struct wire { 
    gate_n * gates; 
    port_n * ports; 
    char * name; 
    int num_gates; 
    int num_ports;
    unsigned int wirelength;  
    unsigned int prev_wirelength;
    unsigned short heir; 
    unsigned short weight;
} wire_t; 
/*
* Layout is a graph. 
* Objects in the graph are gates and ports
* Wires take no space in place personality 
* Wires connect gate/port nodes in graph
*/
typedef struct layout {
    port_t * all_ports; 
    gate_t * all_gates; 
    wire_t * all_wires; 
    gate_t ** grid;
    hier_t * all_hier; 
    int x_size; 
    int y_size; 
    int size_gates; 
    int size_ports; 
    int size_wires;
    unsigned int total_wirelength; 
} layout_t;
/*
* coord in a layout
*/
typedef struct coord {
    int x; 
    int y; 
} coord_t;

int netlist_wireWirelength(layout_t *, wire_n);
int netlist_layoutWirelength(layout_t *); 
void netlist_printNetlist(layout_t*); 
int netlist_wireRevertWirelength(layout_t *, wire_n);
void netlist_printQoR(layout_t *); 
void netlist_free(layout_t *); 
#endif
