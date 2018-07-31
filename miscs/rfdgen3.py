
"""
Created on Mon Sep 18 21:09:26 2017

@author: houben
"""
import numpy as np
np.set_printoptions(threshold=np.inf)
#import csv
#import json
import matplotlib.pyplot as plt
from tkinter import *

#-----tkinter GUI öffnen-----
root = Tk()# blank window
#root.config(height = 1000, width = 1040)
root.title('lin. curve generator')
root.resizable(True, True)


#----- globale Variable, die nachher die Kurve darstellt-----
curve = np.zeros((2, 2))


#----- globale Variable, die nachher die Kurve darstellt-----
# def rfdgensq():
    









#------lineare Aufteilung der Werte über die Zeit-------
def rfdgenlin():
    #Variablen werden aus den Entryfeldern gelesen
    start_t = entry1.get()
    start_t = float(start_t)
    intervall_t = entry2.get()
    intervall_t = float(intervall_t)
    end_t = entry3.get()
    end_t = float(end_t)
    start_v = entry4.get()
    start_v = float(start_v)
    end_v = entry5.get()
    end_v = float(end_v)
    #Prüfung auf Negativität und Wandlung
#    if end_t < 0:
#        ent_t = abs(end_t)
#        start_t = abs(start_t)
#        neg_t = 1
#    if end_v < 0:
#        end_v = abs(end_v)
#        start_v = abs(start_v)
#        neg_v = 1
    global curve
    global count
    count = end_t / float(intervall_t)
    curve = np.zeros((int(count), 2))
    curve[0, 0] = start_t
    curve[0, 1] = start_v
    intervall_v = abs(start_v - end_v) / count
    for ii in range(1,int(count)):
        curve[ii, 0] = start_t + (intervall_t * ii)
        curve[ii, 1] = start_v + (intervall_v * ii)
  
  # file = open('output2.rfd', 'w')
  # file.write(str(curve))
  # file.close()
    root.clipboard_append(np.savetxt(sys.stdout, curve))
    np.savetxt(sys.stdout, curve)
    np.savetxt('your_rfd.rfd', curve)
    #------Kurve wird geplottet-------
    x = np.zeros((int(count),1))
    y = np.zeros((int(count),1))
    for ii in range(0, int(count)):
        x[ii] = curve[ii,0]
        y[ii] = curve[ii,1]
    plt.plot(x,y, label="your .rfd")
    plt.title("Your .rfd")
    plt.xlabel("Zeit")
    plt.ylabel("Wert")
    plt.show()
    return(curve)


#Jason Dump Methode
#json.dump(str(curve), f)
#f.close()

#-----tkinter GUI----- #.place(x=20, y=20)

button1 = Button(root, text="Calculate!", command=rfdgenlin)       

label1 = Label(root, text = "Start X")     
label2 = Label(root, text = "Increment X")   
label3 = Label(root, text = "End X")   
label4 = Label(root, text = "Start Y")   
label5 = Label(root, text = "End Y")                                

entry1 = Entry(root)
entry2 = Entry(root)
entry3 = Entry(root)
entry4 = Entry(root)
entry5 = Entry(root)

button1.pack()
label1.pack(side=TOP)
entry1.pack(side=TOP)
label2.pack(side=TOP)
entry2.pack(side=TOP)
label3.pack(side=TOP)
entry3.pack(side=TOP)
label4.pack(side=TOP)
entry4.pack(side=TOP)
label5.pack(side=TOP)
entry5.pack(side=TOP)

root.mainloop()  
#-----GUI ENDE-----




