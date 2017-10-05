import OpenBCI_Python.open_bci_ganglion as bci
import numpy as np
import pickle
from xgboost.sklearn import XGBClassifier

# Constants
STACK_SIZE = 400 # size of window in samples
UPDATE_INT = 200 # number of samples between updates
USED_CHANNELS = [0,1] # channels range from 0 to 3
LABEL_NAME = ["NEUTRAL", "FORWARD", "LEFT", "RIGHT"]
MODEL_FILE = "./Profiles/{}_model.pkl"
RAW_FILE = "./Profiles/{}_raw.pkl"

# Variables
dataStack = []
rawData = []
freqSpec = []
board = 0
model = 0
connected = False

response = 0

def update_data():
    global rawData, freqSpec
    rawData = np.array(dataStack).T
    freqSpec = np.absolute(np.fft.fft(rawData))

def record_sample(sample):
    global dataStack, freqSpec, model
    stackLen = 0
    x_vect = []

    dataStack.append(sample.channel_data)
    record_sample.count += 1

    if (record_sample.count == STACK_SIZE):
        record_sample.count = 0
        update_data()
        dataStack = []
        board.stop()

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

def train_model(input_vect, label):
    # Edit this to change the ML algorithm
    x_vect = []

    for sample in input_vect:

        tmp = feature_selection(sample)

        if (len(x_vect) == 0):
            x_vect = tmp
        else:
            x_vect = np.vstack((x_vect,tmp))

    tmpModel = XGBClassifier()
    tmpModel.fit(x_vect,label)
    return tmpModel 

def predict(model, input_vect):
    # Edit this too
    tmp = feature_selection(input_vect)
    return model.predict([tmp])

def classify_sample(sample):
    global dataStack, freqSpec, model
    stackLen = 0
    x_vect = []

    dataStack.append(sample.channel_data)
    classify_sample.count += 1

    if (classify_sample.count == UPDATE_INT):
        classify_sample.count = 0
        stackLen = len(dataStack)
        if (stackLen >= STACK_SIZE): 
            # cut dataStack down to STACK_SIZE
            dataStack = dataStack[stackLen-STACK_SIZE:]
            update_data()
            print(predict(model,freqSpec))


def record_data(board, profile, overwrite):
    global freqSpec
    response = 0
    label = 0
    n = 0
    temp = []
    trainData = []
    x_vect = []
    y_vect = []

    if (not overwrite):
        with open(RAW_FILE.format(profile),'rb') as rawFile:
            trainData = pickle.load(rawFile)
            x_vect, y_vect = trainData

    while True:
        label = input("Select label [0-3]: ")
        if (not label.isdigit() or int(label) < 0 or int(label) > 3):
            print("Invalid label.")
            continue
        n = input("How many times: ")
        if (not n.isdigit() or int(n) < 0):
            print("invalid repetitions.")
            continue
        if (int(n) == 0):
            print("Returning to main menu...")
            break

        label = int(label)
        n = int(n)

        for i in range(n): # record data and save into profiles
            print("Dataset {} for {}".format(i, LABEL_NAME[label]))
            input("Press Enter to start.")
            board.start_streaming(record_sample)
            temp = []
            x_vect.append(freqSpec)
            y_vect.append(label)

        print("Record more datasets? [Y/N]")
        response = input("> ")
        if (not response.lower() == 'y'):
            break

    # train + save model and data
    print("Training model...")
    model = train_model(x_vect,y_vect)
    print("Saving...")
    with open(MODEL_FILE.format(profile),'wb+') as modelFile:
        pickle.dump(model, modelFile)''
    with open(RAW_FILE.format(profile),'wb+') as rawFile:
        pickle.dump((x_vect,y_vect), rawFile)

# inits
record_sample.count = 0
classify_sample.count = 0

# main program
while (not connected):
    try:
        board = bci.OpenBCIBoard()
        connected = True
    except OSError:
        input("Press Enter to retry connection.")

while True: # menu hell
    print("1. Record and train model")
    print("2. Online classification")
    print("0. Exit")
    response = input("> ")

    if (response == '1'): # record/train
        print("1. Create/overwrite profile")
        print("2. Add data to profile")
        response = input("> ")

        if (response == '1'): # overwrite
            response = input("Profile name: ")
            record_data(board,response, True)
        elif (response == '2'): # add data
            response = input("Profile name: ")
            try:
                record_data(board,response, False)
            except FileNotFoundError:
                print("Invalid profile.")
                continue
            
    elif (response == '2'): # classify
        response = input("load from profile: ")
        try:
            with open(MODEL_FILE.format(response),'rb') as modelFile:
                model = pickle.load(modelFile)
        except FileNotFoundError:
            print("Invalid profile.")
            continue
        try:
            board.start_streaming(classify_sample)
        except KeyboardInterrupt:
            board.stop()
            print("Stream stopped.")

    elif (response == '0'): # quit
        break
    else:
        print("Invalid input.")

print("Exiting...")
board.disconnect()


