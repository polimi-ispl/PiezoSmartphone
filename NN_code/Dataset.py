import argparse
from scipy import signal
from scipy.io import wavfile
import numpy as np
import struct
import os
import csv


def importPath(Directory, position, i):
    P_input = os.path.join(Directory, i[0])
    P_input = os.path.join(P_input, i[0])
    P_input = P_input + '.wav'

    P_output = os.path.join(Directory, i[0])
    P_output = os.path.join(P_output, position)
    P_output = os.path.join(P_output, 'T/spk_out')
    P_output = os.path.join(P_output, str(i[1]))
    P_output = os.path.join(P_output + '_dbFS.wav')
    return P_input, P_output

def scale_out(y, path):
    a=0
    fin = open(path, "rb")
    while a != b'APx5':
        a=fin.read(4)
    fin.read(4)
    scale=struct.unpack('d', fin.read(8))
    y=y*scale[0]
    fin.close()
    return y, scale[0]

def filter_out(y, filter):
    x = np.load(filter)
    [b, a] = [x.f.ba[0], x.f.ba[1]]
    y = signal.lfilter(b, a, y, axis=-1, zi=None)
    return y

def scale_in(x, db):
    x=x*pow(10, -db/20)
    return x

def main(args):
    Directory = '' #Measurement Directory

    os.makedirs(args.dataFolder, exist_ok=True)
    saveDir = os.path.dirname(os.path.realpath(__file__))
    saveDir = os.path.join(saveDir, args.dataFolder)



    position = '10_C_C'
    list_amplitudes = [0, 3, 6, 10, 20, 40]
    list_of_types = ['Instrument', 'Speech']
    list_of_file =[]
    combination=[]
    for i, type in enumerate(list_of_types):
        list_of_file.append([name for name in os.listdir(os.path.join(Directory, list_of_types[i])) if os.path.isdir(os.path.join(os.path.join(Directory, list_of_types[i]), name))])
        couples=[(type, a[0], a[1])for a in [(f, s) for f in list_of_file[i] for s in list_amplitudes]]
        for b in couples:
            combination.append(b)

    combination_exist = []
    combination_not_exist = []
    for n in combination:
        i=n[1:]
        Directory_type = os.path.join(Directory, n[0])
        P_input, P_output = importPath(Directory_type, position, i)
        if os.path.exists(P_input) & os.path.exists(P_output):
            combination_exist.append(n)
        else:
            combination_not_exist.append(n)

    combination = combination_exist



    csv_file = os.path.join(saveDir, 'data.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        pass

    for n in combination:
        i=n[1:]
        Directory_type = os.path.join(Directory, n[0])
        P_input, P_output = importPath(Directory_type, position, i)
        print(P_output)
        in_rate, in_data_s = wavfile.read(P_input)
        out_rate, out_data_s = wavfile.read(P_output)
        if np.shape(np.shape(in_data_s))==(1,):
            if len(out_data_s) < len(in_data_s):
                out_data_s = np.concatenate((out_data_s, np.zeros(len(in_data_s) - len(out_data_s))))
            if len(out_data_s) > len(in_data_s):
                in_data_s = np.concatenate((in_data_s, np.zeros(len(out_data_s) - len(in_data_s))))

            in_data_s = in_data_s / 2147483392

            out_data_s, scale = scale_out(out_data_s, P_output)
            out_data_s = filter_out(out_data_s, "Elliptic.npz")  # (HighPass 200Hz)
            out_data_s = out_data_s / 2147483392


            exp_in, exp_out= importPath(saveDir, position, i)
            Path = os.path.split(exp_in)[0]
            a, csv_row = importPath(os.path.join(os.path.split(saveDir)[1], n[0]), position, i)
            csv_row = os.path.splitext(csv_row)[0]
            csv_row = os.path.split(csv_row)[1]
            db=csv_row
            csv_row = os.path.join(Path, csv_row)



            Path = os.path.join(Path, db)
            Path_in = os.path.join(Path, 'x')
            Path_out = os.path.join(Path, 'y')

            os.makedirs(Path_in, exist_ok=True)
            os.makedirs(Path_out, exist_ok=True)

            db=int(db.split("_dbFS")[0])

            len_w= 10*48000
            a=0
            while a < len(out_data_s)-len_w:
                ind = int(a/int(len_w/2))
                csv_row2 = os.path.join(csv_row, str(ind))
                with open(csv_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_row2)

                input = scale_in(in_data_s[a:a+len_w], db)
                output = out_data_s[a:a+len_w]
                np.save(os.path.join(Path_in, str(ind)), input)
                np.save(os.path.join(Path_out, str(ind)), output)
                a=a+int(len_w/2)
        else:
            print("error" +P_input)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataFolder", default="") #Directory destination
    args = parser.parse_args()
    main(args)
