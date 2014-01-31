typedef int wire_n; 
typedef struct port { 
    char * name; 
    wire_n wire; 
    int x; 
    int y;  
} port_t; 
typedef struct gate {
    char * name; 
    wire_n fanin; 
    wire_n fanout; 
    int x; 
    int y;
} gate_t; 
typedef struct wire { 
    gate_t * gates; 
    port_t * ports; 
    char * name; 
    int num_gates; 
    int num_ports; 
} wire_t; 
typedef struct layout {
    port_t * all_ports; 
    gate_t * all_gates; 
    wire_t * all_wires; 
    int x_size; 
    int y_size; 
    int size_gates; 
    int size_ports; 
    int size_wires; 
} layout_t;

