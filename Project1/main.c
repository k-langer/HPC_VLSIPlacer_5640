#include "netlist.h"
#include "parser.h"
#include "common.h" 

int main() {
    FILE * netlist_f;
    layout_t * layout = calloc(1,sizeof(layout_t));
    netlist_f = fopen("netlist.txt","r");
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
                default:
                    printf("Something went wrong in parser land"); 
            }
            printf("%s\n",tokens[0]);
        }
    } else {
        printf ("No valid netlist found\n");
        return -1; 
    }
    int i  = 0;
    for (i = 0 ; i < layout->size_gates; i++) {
        printf("%s\n\tFanout: %d\n",layout->all_gates[i].name, 
            layout->all_gates[i].fanout);
        for (int j =0; j < layout->all_gates[i].fanin_size; j++) {
            printf("\tFanin: %d\n",layout->all_gates[i].fanin[j]);
        }
    }
    return 0;
}
