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
    assert(layout->all_ports);
    assert(layout->all_wires);
    assert(layout->all_gates);
    layout->size_ports = 0; 
    layout->size_gates = 0; 
    layout->size_wires = 0; 
    return layout;
}
layout_t* parser_addGate(layout_t *layout, char ** init) {
    return layout; 
}
layout_t* parser_addPort(layout_t *layout, char ** init) {
    return layout; 
}
layout_t* parser_addWire(layout_t *layout, char ** init) {
    return layout; 
}
