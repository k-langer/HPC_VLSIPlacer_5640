#!/usr/bin/python
# -*- coding: utf-8 -*-

from Tkinter import Tk, Canvas, Frame, BOTH, W
import re, sys, os

scale_c = 40 
margin =40

class Draw_Layout(Frame):
  
    def __init__(self, parent,grid,gates,ports):
        Frame.__init__(self, parent)   
        self.grid = grid
        self.gates = gates
        self.ports = ports
        self.parent = parent        
        self.initUI()
        
    def initUI(self):
      
        self.parent.title("Layout")        
        self.pack(fill=BOTH, expand=1)

        x_size = self.grid[0]*scale_c+margin
        y_size = self.grid[1]*scale_c+margin
        canvas = Canvas(self)
        canvas.create_rectangle(margin, margin, \
            (x_size), (y_size), \
            outline="#fff", fill="#fff")

        for gate in self.gates:
            bry = y_size - gate[2]*scale_c
            tly = bry - scale_c
            brx = gate[1]*scale_c+margin
            tlx = brx + scale_c                
            canvas.create_rectangle(tlx, tly, \
                brx, bry, \
                outline="#fb0", fill="#f10")
            canvas.create_text(brx+scale_c/10,(bry+tly)/2, \
                anchor=W,font="Helvetica",\
                text=gate[0])
        for port in self.ports:
            bry = y_size - port[2]*scale_c
            tly = bry - scale_c
            brx = port[1]*scale_c+margin
            tlx = brx + scale_c                
            canvas.create_rectangle(tlx, tly, brx, bry,
                outline="black", fill="green")
            canvas.create_text(brx+scale_c/10,(bry+tly)/2, \
                anchor=W,font="Helvetica",\
                text=port[0])
            
        canvas.pack(fill=BOTH, expand=1)


def main():
    x_size = 0 
    y_size = 0
    gates = []
    ports = [] 
    root = Tk()
    layout_fl = open(os.path.dirname(os.path.abspath(__file__)) \
        + "/layout.txt","r")
    for ln in layout_fl:
        is_match = re.match(r"x_size(\d+)y_size(\d+)",ln)
        if (is_match):
            x_size = int(is_match.group(1))
            y_size = int(is_match.group(2))
        is_match = re.match(r"gatename(.+)x(\d+)y(\d+)",ln)
        if(is_match): 
            gate = is_match.group(1),\
            int(is_match.group(2)),\
            int(is_match.group(3))
            gates.append(gate)
        is_match = re.match(r"portname(.+)x(\d+)y(\d+)",ln)
        if(is_match): 
            port = is_match.group(1),\
            int(is_match.group(2)),\
            int(is_match.group(3))
            ports.append(port) 
    sz = ""
    sz = str(x_size*scale_c+2*margin) + "x" + \
        str(y_size*scale_c+2*margin) 
    grid = x_size,y_size
    ex = Draw_Layout(root,grid,gates,ports)
    root.geometry(sz)
    root.mainloop()  


if __name__ == '__main__':
    main()  
