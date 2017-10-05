import numpy as np
import pickle
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score

# Constants
USED_CHANNELS = [0,1]
MODEL_FILE = "./Profiles/{}_model.pkl"
RAW_FILE = "./Profiles/{}_raw.pkl"

def feature_selection(sample):
    # Edit this to change feature selection
    tmp =[]
    for channel in USED_CHANNELS:

        d, t, a, b = 0,0,0,0

        for i in range(1,9): # delta band
            d += sample[channel][i]
        d = d/8.0

        for i in range(9,15): # theta band
            t += sample[channel][i]
        t = t/6.0

        for i in range(15,29): # alpha band
            a += sample[channel][i]
        a = a/14.0

        for i in range(29,60): # beta band
            b += sample[channel][i]
        b = b/31.0

        tmp = np.hstack((tmp,[d,t,a,b]))

    return tmp

with open(MODEL_FILE.format("test"),'rb') as modelFile:
	model = pickle.load(modelFile)

with open(RAW_FILE.format("test"),'rb') as rawFile:
	raw_x,label = pickle.load(rawFile)

x_vect = []
for sample in raw_x:

    tmp = feature_selection(sample)
    if (len(x_vect) == 0):
        x_vect = tmp
    else:
        x_vect = np.vstack((x_vect,tmp))

print(np.average(cross_val_score(model, x_vect, label,cv=5)))