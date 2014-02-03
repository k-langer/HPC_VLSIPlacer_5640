#include "netlist.h"
#include "common.h"
int netlist_layoutWirelength(layout_t * layout) {
    int sum = 0; 
    for (int i = 0; i < layout->size_wires; i++) {
         sum += netlist_wireWirelength(layout, i); 
    }
    return sum;
}
int netlist_wireWirelength(layout_t * layout, wire_n wiren) {
    wire_t * wire = &(layout->all_wires[wiren]);
    if (wire->num_gates == 0 && wire->num_ports == 0) {
        return 0; 
    }
    int minx = layout->x_size; 
    int maxx = 0; 
    int miny = layout->y_size;
    int maxy = 0;
    int t_x; 
    int t_y; 
    gate_t * tmp_gate; 
    port_t * tmp_port; 
    for (int i = 0; i < wire->num_gates; i++) {
        tmp_gate = &(layout->all_gates[wire->gates[i]]);
        if (!tmp_gate->name) {
            continue; 
        }
        t_x = tmp_gate->x; 
        t_y = tmp_gate->y;
        if (t_x < minx) {
            minx = t_x; 
        } 
        if (t_x > maxx) {
            maxx = t_x; 
        } 
        if (t_y < miny) {
            miny = t_y; 
        } 
        if (t_y > maxy) {
            maxy = t_y;
        }
    }
    for (int i = 0; i < wire->num_ports; i++) {
        tmp_port = &(layout->all_ports[wire->ports[i]]); 
        if (!tmp_port->name) {
            continue;   
        }
        t_x = tmp_port->x; 
        t_y = tmp_port->y; 
        if (t_x < minx) {
            minx = t_x; 
        } 
        if (t_x > maxx) {
            maxx = t_x; 
        } 
        if (t_y < miny) {
            miny = t_y; 
        } 
        if (t_y > maxy) {
            maxy = t_y;
        }
    }     
    int sum = (maxx-minx) + (maxy-miny);
    wire->wirelength = sum;
    return sum; 
}
