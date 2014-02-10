module adder ( gnd, vdd, A, Y1);

input gnd, vdd;
output Y1;
input [64:0] A;

	BUFX2 BUFX2_1 ( .gnd(gnd), .vdd(vdd), .A(Y1_RAW), .Y(Y1) );
	NOR2X1 NOR2X1_1 ( .gnd(gnd), .vdd(vdd), .A(A[33]), .B(A[36]), .Y(_abc_142_n66) );
	NOR2X1 NOR2X1_2 ( .gnd(gnd), .vdd(vdd), .A(A[30]), .B(A[31]), .Y(_abc_142_n67) );
	NAND2X1 NAND2X1_1 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n66), .B(_abc_142_n67), .Y(_abc_142_n68) );
	NOR2X1 NOR2X1_3 ( .gnd(gnd), .vdd(vdd), .A(A[42]), .B(A[43]), .Y(_abc_142_n69) );
	NOR2X1 NOR2X1_4 ( .gnd(gnd), .vdd(vdd), .A(A[37]), .B(A[40]), .Y(_abc_142_n70) );
	NAND2X1 NAND2X1_2 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n69), .B(_abc_142_n70), .Y(_abc_142_n71_1) );
	NOR2X1 NOR2X1_5 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n68), .B(_abc_142_n71_1), .Y(_abc_142_n72_1) );
	NOR2X1 NOR2X1_6 ( .gnd(gnd), .vdd(vdd), .A(A[18]), .B(A[19]), .Y(_abc_142_n73_1) );
	NOR2X1 NOR2X1_7 ( .gnd(gnd), .vdd(vdd), .A(A[13]), .B(A[16]), .Y(_abc_142_n74_1) );
	NAND2X1 NAND2X1_3 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n73_1), .B(_abc_142_n74_1), .Y(_abc_142_n75_1) );
	NOR2X1 NOR2X1_8 ( .gnd(gnd), .vdd(vdd), .A(A[25]), .B(A[28]), .Y(_abc_142_n76_1) );
	NOR2X1 NOR2X1_9 ( .gnd(gnd), .vdd(vdd), .A(A[22]), .B(A[23]), .Y(_abc_142_n77_1) );
	NAND2X1 NAND2X1_4 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n76_1), .B(_abc_142_n77_1), .Y(_abc_142_n78_1) );
	NOR2X1 NOR2X1_10 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n75_1), .B(_abc_142_n78_1), .Y(_abc_142_n79_1) );
	AND2X2 AND2X2_1 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n72_1), .B(_abc_142_n79_1), .Y(_abc_142_n80_1) );
	NOR2X1 NOR2X1_11 ( .gnd(gnd), .vdd(vdd), .A(A[49]), .B(A[52]), .Y(_abc_142_n81_1) );
	NOR2X1 NOR2X1_12 ( .gnd(gnd), .vdd(vdd), .A(A[46]), .B(A[47]), .Y(_abc_142_n82_1) );
	AND2X2 AND2X2_2 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n81_1), .B(_abc_142_n82_1), .Y(_abc_142_n83_1) );
	INVX1 INVX1_1 ( .gnd(gnd), .vdd(vdd), .A(A[55]), .Y(_abc_142_n84_1) );
	NOR2X1 NOR2X1_13 ( .gnd(gnd), .vdd(vdd), .A(A[53]), .B(A[54]), .Y(_abc_142_n85_1) );
	NAND2X1 NAND2X1_5 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n84_1), .B(_abc_142_n85_1), .Y(_abc_142_n86_1) );
	NOR2X1 NOR2X1_14 ( .gnd(gnd), .vdd(vdd), .A(A[50]), .B(A[51]), .Y(_abc_142_n87_1) );
	NOR2X1 NOR2X1_15 ( .gnd(gnd), .vdd(vdd), .A(A[45]), .B(A[48]), .Y(_abc_142_n88_1) );
	NAND2X1 NAND2X1_6 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n87_1), .B(_abc_142_n88_1), .Y(_abc_142_n89_1) );
	NOR2X1 NOR2X1_16 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n86_1), .B(_abc_142_n89_1), .Y(_abc_142_n90_1) );
	NAND2X1 NAND2X1_7 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n83_1), .B(_abc_142_n90_1), .Y(_abc_142_n91_1) );
	NOR2X1 NOR2X1_17 ( .gnd(gnd), .vdd(vdd), .A(A[34]), .B(A[35]), .Y(_abc_142_n92_1) );
	NOR2X1 NOR2X1_18 ( .gnd(gnd), .vdd(vdd), .A(A[29]), .B(A[32]), .Y(_abc_142_n93_1) );
	NAND2X1 NAND2X1_8 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n92_1), .B(_abc_142_n93_1), .Y(_abc_142_n94_1) );
	NOR2X1 NOR2X1_19 ( .gnd(gnd), .vdd(vdd), .A(A[41]), .B(A[44]), .Y(_abc_142_n95_1) );
	NOR2X1 NOR2X1_20 ( .gnd(gnd), .vdd(vdd), .A(A[38]), .B(A[39]), .Y(_abc_142_n96_1) );
	NAND2X1 NAND2X1_9 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n95_1), .B(_abc_142_n96_1), .Y(_abc_142_n97_1) );
	NOR2X1 NOR2X1_21 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n94_1), .B(_abc_142_n97_1), .Y(_abc_142_n98_1) );
	NOR2X1 NOR2X1_22 ( .gnd(gnd), .vdd(vdd), .A(A[17]), .B(A[20]), .Y(_abc_142_n99_1) );
	NOR2X1 NOR2X1_23 ( .gnd(gnd), .vdd(vdd), .A(A[14]), .B(A[15]), .Y(_abc_142_n100_1) );
	NAND2X1 NAND2X1_10 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n99_1), .B(_abc_142_n100_1), .Y(_abc_142_n101_1) );
	NOR2X1 NOR2X1_24 ( .gnd(gnd), .vdd(vdd), .A(A[26]), .B(A[27]), .Y(_abc_142_n102_1) );
	NOR2X1 NOR2X1_25 ( .gnd(gnd), .vdd(vdd), .A(A[21]), .B(A[24]), .Y(_abc_142_n103_1) );
	NAND2X1 NAND2X1_11 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n102_1), .B(_abc_142_n103_1), .Y(_abc_142_n104_1) );
	NOR2X1 NOR2X1_26 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n101_1), .B(_abc_142_n104_1), .Y(_abc_142_n105_1) );
	NAND2X1 NAND2X1_12 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n98_1), .B(_abc_142_n105_1), .Y(_abc_142_n106_1) );
	NOR2X1 NOR2X1_27 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n91_1), .B(_abc_142_n106_1), .Y(_abc_142_n107_1) );
	NOR2X1 NOR2X1_28 ( .gnd(gnd), .vdd(vdd), .A(A[2]), .B(A[3]), .Y(_abc_142_n108_1) );
	NOR2X1 NOR2X1_29 ( .gnd(gnd), .vdd(vdd), .A(A[62]), .B(A[0]), .Y(_abc_142_n109_1) );
	NAND2X1 NAND2X1_13 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n108_1), .B(_abc_142_n109_1), .Y(_abc_142_n110_1) );
	NOR2X1 NOR2X1_30 ( .gnd(gnd), .vdd(vdd), .A(A[9]), .B(A[12]), .Y(_abc_142_n111_1) );
	NOR2X1 NOR2X1_31 ( .gnd(gnd), .vdd(vdd), .A(A[6]), .B(A[7]), .Y(_abc_142_n112_1) );
	NAND2X1 NAND2X1_14 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n111_1), .B(_abc_142_n112_1), .Y(_abc_142_n113_1) );
	OR2X2 OR2X2_1 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n110_1), .B(_abc_142_n113_1), .Y(_abc_142_n114_1) );
	OR2X2 OR2X2_2 ( .gnd(gnd), .vdd(vdd), .A(A[56]), .B(A[57]), .Y(_abc_142_n115_1) );
	NOR2X1 NOR2X1_32 ( .gnd(gnd), .vdd(vdd), .A(A[58]), .B(A[59]), .Y(_abc_142_n116_1) );
	NOR2X1 NOR2X1_33 ( .gnd(gnd), .vdd(vdd), .A(A[60]), .B(A[61]), .Y(_abc_142_n117_1) );
	NAND2X1 NAND2X1_15 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n116_1), .B(_abc_142_n117_1), .Y(_abc_142_n118_1) );
	NOR2X1 NOR2X1_34 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n115_1), .B(_abc_142_n118_1), .Y(_abc_142_n119_1) );
	NOR2X1 NOR2X1_35 ( .gnd(gnd), .vdd(vdd), .A(A[1]), .B(A[4]), .Y(_abc_142_n120_1) );
	NOR2X1 NOR2X1_36 ( .gnd(gnd), .vdd(vdd), .A(A[63]), .B(A[64]), .Y(_abc_142_n121_1) );
	NAND2X1 NAND2X1_16 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n120_1), .B(_abc_142_n121_1), .Y(_abc_142_n122_1) );
	NOR2X1 NOR2X1_37 ( .gnd(gnd), .vdd(vdd), .A(A[10]), .B(A[11]), .Y(_abc_142_n123_1) );
	NOR2X1 NOR2X1_38 ( .gnd(gnd), .vdd(vdd), .A(A[5]), .B(A[8]), .Y(_abc_142_n124_1) );
	NAND2X1 NAND2X1_17 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n123_1), .B(_abc_142_n124_1), .Y(_abc_142_n125_1) );
	NOR2X1 NOR2X1_39 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n122_1), .B(_abc_142_n125_1), .Y(_abc_142_n126_1) );
	NAND2X1 NAND2X1_18 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n119_1), .B(_abc_142_n126_1), .Y(_abc_142_n127_1) );
	NOR2X1 NOR2X1_40 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n114_1), .B(_abc_142_n127_1), .Y(_abc_142_n128_1) );
	AND2X2 AND2X2_3 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n107_1), .B(_abc_142_n128_1), .Y(_abc_142_n129) );
	NAND2X1 NAND2X1_19 ( .gnd(gnd), .vdd(vdd), .A(_abc_142_n80_1), .B(_abc_142_n129), .Y(Y1_RAW) );
endmodule
