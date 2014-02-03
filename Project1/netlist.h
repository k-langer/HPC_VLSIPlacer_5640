typedef int wire_n; 
typedef int gate_n;
typedef int port_n; 

typedef struct port { 
    char * name; 
    wire_n wire; 
    int x; 
    int y;  
} port_t; 
typedef struct gate {
    char * name; 
    wire_n * fanin; 
    wire_n fanout; 
    int fanin_size; 
    int x; 
    int y;
} gate_t; 
typedef struct wire { 
    gate_n * gates; 
    port_n * ports; 
    char * name; 
    int num_gates; 
    int num_ports;
    float wirelength;  
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
    int x_size; 
    int y_size; 
    int size_gates; 
    int size_ports; 
    int size_wires; 
    gate_t ** grid;
} layout_t;

