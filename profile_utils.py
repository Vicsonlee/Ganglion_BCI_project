import sys
import pickle

import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score

from BCI_GUI import USED_CHANNELS, feature_selection

MODEL_FILE = "./Profiles/{}_model.pkl"
RAW_FILE = "./Profiles/{}_raw.pkl"
PROFILE_FILE = "./Profiles/{}_profile.pkl"

def help():
    print("== Help file for profile_utils ==")
    print("run \" python profile_utils.py '<function>', <args> \" to access the utilities")
    print("Utility list:")
    print("convert - args: list of old profiles to convert")
    print("merge - args: output profile name, list of profiles to merge")
    print("stats - args: list of profiles to display stats for")

def convert(*profiles):
    """ Convert old profile formats to the new format """

    for profile in profiles:
        print("Converting " + profile + "...")

        try:
            with open(MODEL_FILE.format(profile),'rb') as modelFile:
                model = pickle.load(modelFile)

            with open(RAW_FILE.format(profile),'rb') as rawFile:
                raw_x,raw_y = pickle.load(rawFile)

            with open(PROFILE_FILE.format(profile),'wb+') as profileFile:
                pickle.dump((raw_x,raw_y, model), profileFile)

        except FileNotFoundError:
            print("File error.")

def merge(output, *profiles):
    """ merge profile raw data, the model needs to be generated again. """
    raw_x, raw_y = [],[]
    model = 0

    for profile in profiles:
        print("Reading " + profile + "...")
        try:
            with open(PROFILE_FILE.format(profile),'rb') as profileFile:
                x, y, model = pickle.load(profileFile)
            raw_x += x
            raw_y += y
        except FileNotFoundError:
            print("File error.")

    with open(PROFILE_FILE.format(output),'wb+') as outFile:
        pickle.dump((raw_x,raw_y, model), outFile)

def stats(*profiles):
    """ Get various stats for profiles """
    for profile in profiles:
        print("Reading " + profile + "...")
        try:
            with open(PROFILE_FILE.format(profile),'rb') as profileFile:
                x, y, model = pickle.load(profileFile)

            print("Number of samples: {},{}".format(len(x),len(y)))

            x_vect = []
            for sample in x:
                tmp = feature_selection(sample)
                if (len(x_vect) == 0):
                    x_vect = tmp
                else:
                    x_vect = np.vstack((x_vect,tmp))
            print("5-fold cross validation score: {}".format(
                np.average(cross_val_score(model, x_vect, y,cv=5))))

        except FileNotFoundError:
            print("File error.")

def main():
    arg_count = len(sys.argv)
    if arg_count <= 1:
        help()
    else:
        arglist = sys.argv[1].split(',')
        util = arglist[0]
        arglist = arglist[1:]

        if util == "convert":
            convert(*arglist)
        elif util == "merge":
            merge(*arglist)
        elif util == "info":
            info(*arglist)
        elif util == "stats":
            stats(*arglist)

if __name__ == "__main__":
    main()