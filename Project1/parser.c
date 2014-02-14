#include "netlist.h"
#include "parser.h"
#include "common.h"
/*
* Given a file name of a netlist, take a parse that netlist 
* into a layout_t data structure. The layout is a block of 
* gates, wires, and ports that are linked together in a graph
* Also contains layout metadata such as size.
* 
* Note: file type of input netlist is my own, but I wrote a perl
* script to translate any structural verilog file into my netlist
* format. This is in the /verilog/ folder
*/
layout_t * parser_parseNetlist(char * netlist_n) {
    FILE * netlist_f;
    layout_t * layout = calloc(1,sizeof(layout_t));
    netlist_f = fopen(netlist_n,"r");
    if (netlist_f) {
        char *buffer = 0; 
        size_t buflen = 0; 
        char ** tokens; 
        getline(&buffer, &buflen, netlist_f);
        tokens = parser_split(buffer, ' ');
        parser_createLayout(layout, tokens);
        while (getline(&buffer, &buflen, netlist_f) != -1) {
            tokens = parser_split(buffer, ' ');
            switch (tokens[0][0]) {
                case 'p':
                    parser_addPort(layout,tokens);
                    break;
                case 'g':
                    parser_addGate(layout,tokens); 
                    break;
                case '#':
                    break;
                default:
                    printf("Something went wrong in parser land"); 
            }
        }
    } else {
        printf ("No valid netlist found\n");
        return NULL; 
    }
    return layout;
}
/* 
* Called on the first line of parsing, provides a layout construct
* Layout construct (layout_t) is a structure that servers the purpose
* of a layout class in an OOP
* Holds all gates/ports/wires
* Also manages the grid that these objects live in
* Will likely be exported as LEF/DEF someday
*/ 
layout_t* parser_createLayout(layout_t * layout, char ** init) {
    layout->x_size = atoi(init[0]);
    layout->y_size = atoi(init[1]);
    layout->size_wires = atoi(init[2]); 
    layout->size_gates = atoi(init[3]); 
    layout->size_ports = atoi(init[4]);
    layout->all_ports = calloc(layout->size_ports, sizeof(port_t)); 
    layout->all_gates = calloc(layout->size_gates, sizeof(gate_t)); 
    layout->all_wires = calloc(layout->size_wires, sizeof(wire_t)); 
    int i; 
    for (i = 0; i < layout->size_wires; i++) {
        char ** buffer= &(layout->all_wires[i].name);
        *buffer = malloc(i/10+2); 
        sprintf(*buffer,"%d",i);
    } 
    layout->grid = calloc(layout->x_size*layout->y_size,sizeof(gate_t *));
    assert(layout->all_ports);
    assert(layout->all_wires);
    assert(layout->all_gates);
    layout->size_ports = 0; 
    layout->size_gates = 0; 
    return layout;
}
/* 
* Link gates to wires in layout graph
*/
void parser_linkGate(layout_t *layout, wire_n wiren, gate_n gaten) {
    wire_t * wire = &(layout->all_wires[wiren]); 
    int sz = wire->num_gates + 1; 
    gate_n * new_gate_bfr = malloc(sizeof(gate_n) * sz);  
    //memcpy of 0 bytes is actually undefined behavior
    //But I can't see why std libs would break this
    memcpy(new_gate_bfr,wire->gates,sizeof(gate_n) * (sz-1)); 
    new_gate_bfr[sz-1] = gaten; 
    if(wire->gates) {
        free(wire->gates);
    }
    wire->gates = new_gate_bfr;
    wire->num_gates += 1;  
}
/*
* Parse any line that has gates
* add to layout and link gates to wires
*/
layout_t* parser_addGate(layout_t *layout, char ** init) {
    int i = 1; 
    int fanin_size = 0; 
    wire_n * fanin_bfr = malloc(sizeof(wire_n)*20);
    gate_t * self_ptr = &(layout->all_gates[layout->size_gates]); 
    while (init[i]) { 
        if (strstr(init[i],"name=")) {
            int t_len = strlen("name="); 
            int size = strlen(init[i]) - t_len; 
            char ** t_ptr = &(self_ptr->name);
            *t_ptr = calloc(sizeof(char),size); 
            memcpy(*t_ptr, init[i] + t_len, size);  
        } else if (strstr(init[i],"fanin=")) {
            assert(fanin_size < 20); 
            int t_len = strlen("fanin="); 
            fanin_bfr[fanin_size] = atoi(init[i]+t_len);
            parser_linkGate(layout,fanin_bfr[fanin_size],layout->size_gates); 
            fanin_size++;
        } else if (strstr(init[i],"fanout=")) {
            int t_len = strlen("fanout="); 
            wire_n * fi = &(self_ptr->fanout);
            *fi = atoi(init[i]+t_len);
            parser_linkGate(layout,*fi,layout->size_gates); 
        } 
        i++;
    }
    self_ptr->fanin = malloc(sizeof(wire_n)*fanin_size); 
    self_ptr->fanin_size = fanin_size;
    memcpy(self_ptr->fanin, fanin_bfr, fanin_size * sizeof(wire_n));
    free(fanin_bfr);
    layout->size_gates += 1;
    return layout; 
}
/* Given a newly created port, link it to its wire in the layout
*/
void parser_linkPort(layout_t *layout, wire_n wiren, port_n portn) {
    wire_t * wire = &(layout->all_wires[wiren]); 
    int sz = wire->num_ports + 1; 
    port_n * new_port_bfr = malloc(sizeof(port_n) * sz);  
    //memcpy of 0 bytes is actually undefined behavior
    //But I can't see why std libs would break this
    memcpy(new_port_bfr,wire->ports,sizeof(port_n) * (sz-1)); 
    new_port_bfr[sz-1] = portn;
    if (wire->ports) {
        free(wire->ports);
    } 
    wire->ports = new_port_bfr;
    wire->num_ports += 1;  
}
/*Given a line that specifies a port add it to the netlist
*/
layout_t* parser_addPort(layout_t *layout, char ** init) {
    int i = 1; 
    port_t * self_ptr = &(layout->all_ports[layout->size_ports]); 
    while (init[i]) { 
        if (strstr(init[i],"name=")) {
            int t_len = strlen("name="); 
            int size = strlen(init[i]) - t_len; 
            char ** t_ptr = &(self_ptr->name);
            *t_ptr = malloc(size); 
            memcpy(*t_ptr, init[i] + t_len, size);  
        } else if (strstr(init[i],"x=")) {
            int x_len = strlen("x="); 
            int * x = &(self_ptr->x);
            *x = atoi(init[i]+x_len);
        } else if (strstr(init[i],"y=")) {
            int y_len = strlen("y="); 
            int * y = &(self_ptr->y);
            *y = atoi(init[i]+y_len);
        } else if (strstr(init[i],"wire=")) {
            int t_len = strlen("wire="); 
            wire_n * fi = &(self_ptr->wire);
            *fi = atoi(init[i]+t_len);
            parser_linkPort(layout,*fi,layout->size_ports); 
        } 
        i++;
    }
    layout->size_ports += 1;
    return layout; 
}
/*
*TODO: allow for wires to take on names
*/
layout_t* parser_addWire(layout_t *layout, char ** init) {
    return layout; 
}
/*str_spit shamelessly stolen from 
*http://stackoverflow.com/questions/9210528/split-string-with-delimiters-in-c
*/
char** parser_split(char* a_str, const char a_delim)
{
    char** result    = 0;
    size_t count     = 0;
    char* tmp        = a_str;
    char* last_delim = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;
    while (*tmp) {
        if (a_delim == *tmp) {
            count++;
            last_delim = tmp;
        }
        tmp++;
    }
    count += last_delim < (a_str + strlen(a_str) - 1);
    count++;
    result = malloc(sizeof(char*) * count);
    if (result) {
        size_t idx  = 0;
        char* token = strtok(a_str, delim);
        while (token) {
            assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        *(result + idx) = 0;
    }
    return result;
}
