#!/usr/bin/python
# -*- coding: utf-8 -*-

from Tkinter import Tk, Canvas, Frame, BOTH, W
import re, sys, os
scale_c = 6
margin =40
x_origin = 0*scale_c
y_origin = 0*scale_c

class Draw_Layout(Frame):
  
    def __init__(self, parent,grid,gates,ports):
        Frame.__init__(self, parent)   
        self.grid = grid
        self.gates = gates
        self.ports = ports
        self.parent = parent      
        self.draw()
        parent.bind('<Key>',key);
    def draw(self):
        self.parent.title("Layout")        
        self.pack(fill=BOTH, expand=1)

        x_size = self.grid[0]*scale_c+margin
        y_size = self.grid[1]*scale_c+margin
        canvas = Canvas(self,width=800,height=450)
        canvas.create_rectangle(margin+x_origin, margin+y_origin, \
            (x_size+x_origin), (y_size+y_origin), \
            tags="Obj", outline="#fff", fill="#fff")
        for gate in self.gates:
            fill_t = "#f10"
            if (re.match(r"DFF",gate[0])):
                fill_t = "#05f"
            bry = y_size - gate[2]*scale_c+y_origin
            tly = bry - scale_c
            brx = gate[1]*scale_c+margin+x_origin
            tlx = brx + scale_c                
            canvas.create_rectangle(tlx, tly, \
                brx, bry, \
                tags="Obj",outline=fill_t, fill=fill_t)
           # canvas.create_text(brx+scale_c/10,(bry+tly)/2, \
           #     anchor=W,font="Helvetica",\
           #     text=gate[0])
        for port in self.ports:
            y_fix_port = 0; 
            x_fix_port = 0; 
            if (port[2] >= self.grid[1]-1):
                y_fix_port = -scale_c+y_origin
            if (port[2] <= 1):
                y_fix_port = scale_c+y_origin
            if(port[1] >= self.grid[0]-1):
                x_fix_port = scale_c+x_origin
            if(port[1] <= 1):
                x_fix_port = -scale_c+x_origin
            bry = y_size - port[2]*scale_c+y_fix_port
            tly = bry - scale_c 
            brx = port[1]*scale_c+margin+x_fix_port
            tlx = brx + scale_c                
            canvas.create_rectangle(tlx, tly, brx, bry,
                tags="Obj",outline="black", fill="green")
            canvas.create_text(brx+scale_c/10,(bry+tly)/2, \
                anchor=W,font="Helvetica",\
                text=port[0])           
        canvas.pack(fill=BOTH, expand=1)
def key(event):
    x = event.char
    global y_origin
    global scale_c
    global x_origin
    if x == "w":
        for item in canvasE.scene:
            canvasE.scale( item, 0, 0, WIDTH_RATIO, HEIGHT_RATIO )
        y_origin -= scale_c
    if x == "s":
        y_origin += scale_c            
    if x == "a":
        x_origin -= scale_c
    if x == "d":
        x_origin += scale_c
    if x == "z": 
        if (scale_c < 100):
            scale_c *= 2
    if x == "x":
        if (scale_c > 2):
            scale_c /= 2
    print scale_c 
    print y_origin 
    print x_origin 

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
    #sz = "800x450"
    grid = x_size,y_size
    ex = Draw_Layout(root,grid,gates,ports)
    #root.bind('<Key>',key);
    root.geometry(sz)
    root.mainloop()  


if __name__ == '__main__':
    main()  
