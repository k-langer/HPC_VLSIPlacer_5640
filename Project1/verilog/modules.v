module BUFX2 ( gnd, vdd, A, Y);
    input gnd, vdd, A; 
    output Y;
endmodule 

module AND2X2 ( gnd, vdd, A, B, Y);
    input gnd, vdd, A, B;
    output Y;
endmodule 
module DFFPOSX1 (gnd, vdd, CLK, D, Q);
    input gnd, vdd, CLK, D;
    output Q; 
endmodule
module DFFSR (gnd, vdd, CLK, D, Q, R, S);
    input gnd, vdd, CLK, D, R, S;
    output Q; 
endmodule
module INVX1 (gnd, vdd, A, Y);
    input gnd, vdd, A;
    output Y;
endmodule
module INVX2 (gnd, vdd, A, Y);
    input gnd, vdd, A;
    output Y;
endmodule
module INVX4 (gnd, vdd, A, Y);
    input gnd, vdd, A;
    output Y;
endmodule
module INVX8 (gnd, vdd, A, Y);
    input gnd, vdd, A;
    output Y;
endmodule
module NAND2X1 ( gnd, vdd, A, B, Y);
    input gnd, vdd, A, B;
    output Y;
endmodule 
module NOR2X1 ( gnd, vdd, A, B, Y);
    input gnd, vdd, A, B;
    output Y;
endmodule 
module OR2X2 ( gnd, vdd, A, B, Y);
    input gnd, vdd, A, B;
    output Y;
endmodule 
module XNOR2X1 ( gnd, vdd, A, B, Y);
    input gnd, vdd, A, B;
    output Y;
endmodule 
module XOR2X1 ( gnd, vdd, A, B, Y);
    input gnd, vdd, A, B;
    output Y;
endmodule 
