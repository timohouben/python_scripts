# -*- coding: utf-8 -*
#import PyPDF2 as pdf
from PyPDF2 import PdfFileWriter, PdfFileReader
import os
import numpy as np

def merger(output_path, input_paths):
    pdf_writer = PdfFileWriter()

    for path in input_paths:
        pdf_reader = PdfFileReader(path)
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))

    with open(output_path, 'wb') as fh:
        pdf_writer.write(fh)

# der Pfad wo die combinierten pdfs gespeichert werden sollen
home = "/Users/houben/Desktop/leo_pdf"
# der Pdaf zu den pdfs
path = "/Volumes/Elements/Drillbit_Landwuest/specs"
# der Pfad zu der liste.txt inklusive Dateinamen
path_liste = "/Volumes/Elements/Drillbit_Landwuest/leo_pdf/liste.txt"

files = os.listdir(path)

# durch Inhalt (files) von "path" iterieren (ist Inhalt ein Ordner?) Funktion isdir prueft ob Inhalt Ordner ist, Ergebnis wird in Liste directories gespeichert
directories = []
for i in files:
    if os.path.isdir(path + "/" + i) == True:
        directories.append(i)

# durch B2X soll nicht geloopt werden, pass = weitermachen
try:
    directories.remove("B2X")
except ValueError:
    pass

datei = open(path_liste)
sortierung = []

# strip entfernt Leerzeilen/Newlines, es wird durch "datei" (-> liste.txt) iteriert, um die Zeilen in sortierung zu übertragen
for i in datei:
    sortierung.append(i.strip())

finale_sortierung = []
# i Zahl, item Eintrag in sortierung
for i,item in enumerate(sortierung):
    # nur wenn i/3 ohne Rest teilbar ist
    if i%3 == 0:
        # finale_sortierung wird mit 3 Eintraegen pro Zeile sortiert
        finale_sortierung.append([sortierung[i],sortierung[i+1],sortierung[i+2]])

count = 1
for geophon in finale_sortierung:
    list_pre_sorting = []
    print("Starting with geophon " + geophon[0][:-3] + ", " + str(count) + " of " + str(len(finale_sortierung)))
    for components in geophon:
        for dire in directories:
            files = os.listdir(path + "/" + dire)
            try:
                files.remove("{2_}")
            except ValueError:
                pass
            for file in files:
                if components == file[:-15]:
                    list_pre_sorting.append(dire + "/" + file)

    list_post_sorting = sorted(list_pre_sorting,key=lambda x: x[-14:-6])
    list_post_sorting_paths = [path + "/" + i for i in list_post_sorting]
    merger(home + "/" + geophon[0][:-3] + "_merge.pdf", list_post_sorting_paths)
    count = count + 1
print("rdy!")
