#include <stdio.h>
int main() {
    FILE * netlist_f;
    netlist_f = fopen("netlist.txt","r");
    if (netlist_f) {
        char *buffer = 0; 
        size_t buflen = 0; 
        while (getline(&buffer, &buflen, netlist_f)) != -1) {
            printf(buffer);
        }
    } else {
        printf ("No valid netlist found\n");
    }
    return 0;

}
