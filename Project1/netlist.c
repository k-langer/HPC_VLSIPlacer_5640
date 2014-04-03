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
    layout->total_wirelength = sum;
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
        tmp_gate = layout->all_gates + wire->gates[i];
        //if (!tmp_gate->name) {
        //    continue; 
        //}
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
        tmp_port = layout->all_ports + wire->ports[i]; 
        //if (!tmp_port->name) {
        //    continue;   
        //}
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
    if (wire->wirelength == 0) {
        wire->prev_wirelength = sum;
    } else {
        wire->prev_wirelength = wire->wirelength;           
    }
    wire->wirelength = sum;
    return sum; 
}
int netlist_wireRevertWirelength(layout_t * layout, wire_n wiren) {
    /*Profiling shows that most time is spent calculating wirelengths 
    * Implementing roll-back will prevent recalculation 
    * First attempt caused significant problems and is **HIGH** on the do list
    * Note: will be hard to stay thread safe...
    */
    //wire_t * wire_tmp = &(layout->all_wires[wiren]); 
    //int old_wl = wire_tmp->wirelength; 
    //wire_tmp->wirelength = wire_tmp->prev_wirelength; 
    //wire_tmp->prev_wirelength = old_wl;
    netlist_wireWirelength(layout,wiren);
    return 0;
}
/* Dump the netlist into a file 
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
hier_t * getHier(layout_t * layout, short hier_ptr) {
    hier_t * rt = &(layout->all_hier[0]); 
    for (int i = 0; i < hier_ptr; i++) {
        rt = rt->next;
    }
    return rt;
}
void netlist_printForVisualizer(layout_t * layout) {
    for (int i = 0; i < layout->size_gates; i++) {
        float x = (layout->all_gates+i)->x + 0.0f;
        float y = (layout->all_gates+i)->y + 0.0f;
        printf("%d %f %f\n",i,x,y);
    }
}
void netlist_printForMatlab(layout_t * layout) {
    for (int i = 0; i < layout->size_gates; i++) {
        float x = (layout->all_gates+i)->x + 0.0f;
        printf("%f, ",x);
    }
    printf("\n");
    for (int i = 0; i < layout->size_gates; i++) {
        float y = (layout->all_gates+i)->y + 0.0f;
        printf("%f, ",y);
    }
}
void netlist_printQoR(layout_t * layout) {
   hier_t * hier; 
   for (int i = 0; i < layout->size_gates; i++) {
        gate_t * gate_ptr = &(layout->all_gates[i]);
        short hier_ptr = layout->all_wires[gate_ptr->fanout].heir;
        hier = getHier(layout,hier_ptr); 
        int gx = gate_ptr->x; 
        int gy = gate_ptr->y;
        if (gx < hier->x_min) {
            hier->x_min = gx; 
        }
        if (gx > hier->x_max) {
            hier->x_max = gx; 
        } 
        if (gy < hier->y_min) {
            hier->y_min = gy; 
        } 
        if (gy > hier->y_max) {
            hier->y_max = gx; 
        }
    }
    hier = &(layout->all_hier[0]);
    printf ("--Quality of Results--\n");
    double density; 
    while (hier) {
        density = 1;
        if (hier->size > 1) {
            density = (hier->size) / ((hier->x_max - hier->x_min + 0.0)*(hier->y_max - hier->y_min+0.0)); 
        }
        hier->density = density; 
        printf("%s\n\tAvg. Spread (%d , %d)(%d , %d)\n\tSize: %d\n",hier->label, hier->x_min,hier->y_min,hier->x_max,hier->y_max,hier->size);
        hier = hier->next; 
    }
    density = (layout->size_gates)/((layout->x_size+0.0)*(layout->y_size+0.0));
    printf("Total density %.4f\nTotal Wirelength %d\n",density,layout->total_wirelength);
    printf("Total gates: %d\nX bounds: %d\nY bounds: %d\n",layout->size_gates, layout->x_size, layout->y_size); 
    
}
void netlist_free(layout_t * layout) {
    for (int i = 0; i < layout->size_gates; i++) {
        free(layout->all_gates[i].name);
        free(layout->all_gates[i].fanin);
    }
    for (int i = 0; i < layout->size_ports; i++) {
        free(layout->all_ports[i].name);
    }
    for (int i = 0; i < layout->size_wires; i++) {
        free(layout->all_wires[i].gates);
        free(layout->all_wires[i].ports);
        free(layout->all_wires[i].name);
    }
    free(layout->grid); 
    free(layout->all_ports);
    free(layout->all_gates);
    free(layout->all_wires);   
    free(layout); 
}




