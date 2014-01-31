char** parser_split(char*, const char);

layout_t* parser_createLayout(layout_t * , char ** );
layout_t* parser_addGate(layout_t *, char **);
layout_t* parser_addPort(layout_t * , char **);
layout_t* parser_addWire(layout_t *, char **);
