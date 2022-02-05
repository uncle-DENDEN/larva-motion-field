from online_tracking.fillnull import avfill
import joblib
import pandas as pd
import numpy as np
import math

inputfile = r'D:\workspace\tracking\learning\tracking_output\reinforcement 1\output\reinf_4-track.txt'
outputfile = r'D:\workspace\tracking\learning\tracking_output\reinforcement 1\track_filled\reinf_4-track_filled.txt'
centerfile = r'D:\workspace\tracking\learning\tracking_output\reinforcement 1\zone file\centre.csv'
# read output data
se = pd.read_table(inputfile, sep=' ', names=['X_coord', 'Y_coord'])
# load center
center = joblib.load('out_file/t_cod.pkl')
plate_index = 0
ce, thre = center[plate_index]

# write filled data
sep = avfill(se, ce, thre)
sep.to_csv(outputfile, header=None, index=None, sep=' ', mode='a')
