#ifndef __parser__h
#define __parser__h

layout_t *  parser_parseNetlist(char *);
char** parser_split(char*, const char);
void parser_linkGate(layout_t *, wire_n, gate_n);
void parser_linkPort(layout_t *, wire_n, port_n);
layout_t* parser_createLayout(layout_t * , char ** );
layout_t* parser_addGate(layout_t *, char **);
layout_t* parser_addPort(layout_t * , char **);
layout_t* parser_addWire(layout_t *, char **);
#endif
