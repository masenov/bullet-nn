import os
import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab

def getAllFiles():
    directory = 'build_cmake/experiments'
    files = os.listdir(directory)
    return files

def varsFromFile(files, fileNumber):
    file = files[fileNumber]
    rest_pos = 9
    fric_pos = file.find('fric')
    tilt_pos = file.find('tilt')
    mass_pos = file.find('mass')
    end_pos = file.find('.txt')
    rest = float(file[rest_pos:fric_pos])
    fric = float(file[fric_pos+5:tilt_pos])
    tilt = float(file[tilt_pos+4:mass_pos])
    mass = float(file[mass_pos+4:end_pos])
    return (rest,fric,tilt,mass)

def fileFromVars(mass, rest, fric, tilt):
   tilt = tilt*45.0
   filename = "data_rest" + str(rest) + "fric_" + str(fric) + "tilt" + str(tilt)+ "mass"+str(mass)+ ".txt"
   return filename

def loadData(files, fileNumber):
    directory = 'build_cmake/experiments'
    file = files[fileNumber]
    data = np.zeros((1000,10))
    count = 0
    with open(directory + '/' + file, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if (count==1000):
                break
            for i in range(10):
                data[count][i] = float(row[i])
            count+=1
    return data

def Data(start,end):
    train = np.zeros((end-start,1000,10))
    files = getAllFiles()
    vars = varsFromFile(files,1)
    for i in range(end-start):
        print (i)
        train[i,:,:] = loadData(files,start+i)
    return train

def plot(fileNumber, files, filename=None, every=1):
    if filename is None:
        file = files[fileNumber]
    else:
        file = filename
    x = []
    y = []
    z = []
    count = 0
    with open('build_cmake/experiments/' + file, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            count += 1
            if count==every:
                x.append(float(row[0]))
                y.append(float(row[1]))
                z.append(float(row[2]))
                count = 0

    colors = cm.rainbow(np.linspace(0, 1, len(x)))
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')



    # ax.scatter(x, y, z, c=colors)
    # pylab.show()
    fig2 = plt.figure(num=0)
    ax = plt.axes()
    ax.scatter(x,y,c=colors)
    pylab.show()

def clearData():
    with open('data.txt', 'w') as fout:
                    fout.writelines("")



def sequenceData(filename, length):
    count = 0
    data = np.zeros((3,length))
    print (filename)
    with open('build_cmake/experiments/' + filename, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if count < length:
                data[0,count] = float(row[0])
                data[1,count] = float(row[1])
                data[2,count] = float(row[2])
                count += 1
            else:
                with open('data.txt', 'a') as fout:
                    for i in range(length):
                        for j in range(3):
                            fout.writelines(str(data[j,i]) + ', ')
                    for i in range(6):
                        fout.writelines(row[i+3] + ', ')
                    fout.writelines(row[9])
                    fout.writelines('\n')
                data[:,:length-1] = data[:,1:]
                data[0,length-1] = float(row[7])
                data[1,length-1] = float(row[8])
                data[2,length-1] = float(row[9])

def seqData():
    file = 'data.txt'

    num_lines = sum(1 for line in open('data.txt'))

    with open('data.txt', 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            len_data = len(row)
            break

    data = np.zeros((num_lines,len_data))
    count = 0
    with open('data.txt', 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            for i in range(len_data):
                data[count][i] = float(row[i])
            count+=1
    return data


#files = getAllFiles()
#print (len(files))
#print (varsFromFile(files,210))
#print (fileFromVars(0.1,0.1,0.1,0.7))
#for i in range(0,10):
#     plot(210, files, every=1, filename=fileFromVars(0.1,0.1,0.1,i/10.0))
#clearData()
#sequenceData(fileFromVars(0.1,0.1,0.1,0.9),3)
# for rest in range(0,10):
#     for fric in range(0,10):
#         for tilt in range(0,10):
#             sequenceData(fileFromVars(1/10.0,rest/10.0,fric/10.0,tilt/10.0),10)
# data = seqData()
# print (len(data))
