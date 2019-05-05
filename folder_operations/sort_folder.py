import os
import numpy as np
# This script sorts folders based on the first integer before the seperator "_" and moves it to subdirectories. Folder numbers have to be unique!!
# get directory
directory = str(input("Insert directory: "))
#min = str(input("Minimum value to look for: "))
#min = str(input("Maximum value to look for: "))
group = min = int(input("How many files per folder?: "))
folder_list = os.listdir(directory)
folder_list.sort()
# remove files from list
for folder in folder_list:
    if os.path.isdir(directory + "/" + folder) == False:
        folder_list.remove(folder)
# split string
folder_list_split = [i.split("_") for i in folder_list]
# get list of integers before "_"
integer_list = []
for i in np.arange(len(folder_list_split)):
    print(folder_list_split[i][0])
    # remove folders without digits before "_"
    if folder_list_split[i][0].isdigit() == False:
        print("Remove folder ", folder_list[i], "from list..")
        del folder_list_split[i]
        del folder_list[i]
    else:
        integer_list.append(int(folder_list_split[i][0]))
print("Folder list: ", folder_list)
print("Integer List: ", integer_list)
np.savetxt(directory + "/new_directories.txt", integer_list)

# how many new folder
n_new_folders = len(integer_list) // group

# create bins
bins = []
bin_temp = []
for i, item in enumerate(integer_list):
    #bin_temp.append(folder_list[i])
    bin_temp.append(int(item))
    if len(bin_temp) == group:
        bins.append((np.min(bin_temp),np.max(bin_temp)))
        bin_temp = []

# numbers of folders is not integer dividable by group size
if n_new_folders != (len(integer_list) / group):
    bins.append((bins[-1][1]+1,np.max(integer_list)))

# sort folders
for i, item in enumerate(bins):
    new_dir = directory + "/" + str(item[0]) + "-" + str(item[1])
    os.mkdir(new_dir)
    for j, jtem in enumerate(integer_list):
        if jtem >= item[0] and jtem <= item[1]:
            os.rename(directory + "/" + folder_list[j], str(new_dir) + "/" + folder_list[j])
