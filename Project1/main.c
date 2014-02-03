#include "netlist.h"
#include "parser.h"
#include "common.h" 
#include "solver.h" 
#include "annealer.h"

int main() {
    layout_t *layout = parser_parseNetlist("netlist.txt"); 
    if (!layout) {
        return -1;
    }
    annealer_anneal(layout,0);
//    int i  = 0;
    int sum = netlist_layoutWirelength(layout); 
    printf("Wirelength: %d\n",sum); 
    /*
    //Lets test gates
    for (i = 0 ; i < layout->size_gates; i++) {
        printf("%s\n\tFanout: %d\n",layout->all_gates[i].name, 
            layout->all_gates[i].fanout);
        for (int j =0; j < layout->all_gates[i].fanin_size; j++) {
            printf("\tFanin: %d\n",layout->all_gates[i].fanin[j]);
        }
    }
    */ 
    /* 
    //lets test wires
    for (i = 0; i < layout->size_wires; i++) {
        wire_t * wire_ptr = &(layout->all_wires[i]); 
        printf("Wire: %s\n",wire_ptr->name);
        for ( int j = 0; j < wire_ptr->num_gates; j++){
            gate_n gtn = wire_ptr->gates[j];
            printf("\tG: %s\n",layout->all_gates[gtn].name);
        } 
        for ( int j = 0; j < wire_ptr->num_ports; j++){
            port_n gtn = wire_ptr->ports[j];
            printf("\tP: %s\n",layout->all_ports[gtn].name);
        } 
        printf("\tSize: %d\n",wire_ptr->num_gates);
    } 
    */
    /*
    //Lets test ports 
    for (i = 0; i < layout->size_ports; i++) {
        port_t * port_ptr = &(layout->all_ports[i]);
        printf("Port: %s\n\t%d\n",port_ptr->name, port_ptr->wire);
    }
    */
    return 0;
}
