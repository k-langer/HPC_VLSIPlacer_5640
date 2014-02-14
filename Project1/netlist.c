#include "netlist.h"
#include "common.h"
/* Calculate wirelength of whole netlist
*/
int netlist_layoutWirelength(layout_t * layout) {
    int sum = 0; 
    for (int i = 0; i < layout->size_wires; i++) {
        //Wirelengths follow and are recorded by wires
         sum += netlist_wireWirelength(layout, i); 
    }
    return sum;
}
/* Caclulate wirelength of just one wire in the netlist
 * given a wire pointer wiren
 * Wirelengh is just bounding box, and is a gross approximation 
 * that works well for small fan-outs. 
*/
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
        //Calculate gate bounding box
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
        //Calculate port bounding box
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
    //Return and update wirelength 
    int sum = (maxx-minx) + (maxy-miny);
    wire->wirelength = sum;
    return sum; 
}
/* Dumb the netlist into a file 
 * File consumed by GUI
*/
void netlist_printNetlist(layout_t * layout) {
    FILE *fp = fopen("display/layout.txt", "w+");
    fprintf(fp,"x_size%dy_size%d\n",layout->x_size,layout->y_size); 
    gate_t * tmp_gate; 
    port_t * tmp_port; 
    for (int i = 0; i < layout->size_gates; i++) {
        tmp_gate = &(layout->all_gates[i]);
        if (tmp_gate->name) { 
            fprintf(fp,"gatename%sx%dy%d\n"
                ,tmp_gate->name,tmp_gate->x,tmp_gate->y);
        }
    }
    for (int i = 0; i < layout->size_ports; i++) {
        tmp_port = &(layout->all_ports[i]); 
        if (tmp_port->name) {
            fprintf(fp,"portname%sx%dy%d\n",tmp_port->name,tmp_port->x,tmp_port->y);
        }
    }   
}
