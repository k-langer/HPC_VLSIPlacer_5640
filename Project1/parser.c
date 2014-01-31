#include "netlist.h"
#include "parser.h"
#include "common.h"
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
    assert(layout->all_ports);
    assert(layout->all_wires);
    assert(layout->all_gates);
    layout->size_ports = 0; 
    layout->size_gates = 0; 
    return layout;
}

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
            *t_ptr = malloc(size); 
            memcpy(*t_ptr, init[i] + t_len, size);  
        } else if (strstr(init[i],"fanin=")) {
            assert(fanin_size < 20); 
            int t_len = strlen("fanin="); 
            fanin_bfr[fanin_size] = atoi(init[i]+t_len);
            fanin_size++; 
        } else if (strstr(init[i],"fanout=")) {
            int t_len = strlen("fanout="); 
            wire_n * fi = &(self_ptr->fanout);
            *fi = atoi(init[i]+t_len);
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

layout_t* parser_addPort(layout_t *layout, char ** init) {
    return layout; 
}

layout_t* parser_addWire(layout_t *layout, char ** init) {
    return layout; 
}
