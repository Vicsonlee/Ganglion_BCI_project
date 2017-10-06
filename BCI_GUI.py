import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import pickle
import threading
import OpenBCI_Python.open_bci_ganglion as bci
from xgboost.sklearn import XGBClassifier

### Constants ###
STACK_SIZE = 400 # size of window in samples
UPDATE_INT = 200 # number of samples between updates
USED_CHANNELS = [0,1] # channels range from 0 to 3
LABEL_NAME = ["NEUTRAL", "FORWARD", "LEFT", "RIGHT"]
DIR_NAME = ["UP","RIGHT","DOWN","LEFT"]
MODEL_FILE = "./Profiles/{}_model.pkl"
RAW_FILE = "./Profiles/{}_raw.pkl"

### Globals ###
profileName = ''
model = 0
board = 0
raw_x = []
raw_y = []
dataStack = []
rawData = []
freqSpec = []
connected = False
exit = False

def handle_sample(sample):
    global board, dataStack, exit

    if exit:
        # IT'S TIME TO STOP
        board.stop()
        return

    stackLen = len(dataStack)
    dataStack.append(sample.channel_data)

    if (stackLen > STACK_SIZE): 
        # cut dataStack down to STACK_SIZE
        dataStack = dataStack[stackLen-STACK_SIZE:]

def update_data():
    global rawData, freqSpec, dataStack
    stackLen = len(dataStack)
    if (stackLen > STACK_SIZE): 
        # cut dataStack down to STACK_SIZE
        dataStack = dataStack[stackLen-STACK_SIZE:]
    rawData = np.array(dataStack).T
    freqSpec = np.absolute(np.fft.fft(rawData))

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
    return model.predict([tmp])[0]

### Classes ###

class GanglionThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        global board, connected, exit
        while (not connected and not exit):
            try:
                board = bci.OpenBCIBoard()
                connected = True
            except (OSError, ValueError):
                print("Retrying...")
        if not exit:
            board.start_streaming(handle_sample)

### GUI classes begin here ###
class ProfilePage(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self,parent)
        self.parent = parent

        buttonframe = tk.Frame(self)
        entryframe = tk.Frame(buttonframe)

        label = tk.Label(entryframe, text="Profile: ")
        self.entry = tk.Entry(entryframe)

        label.pack(side="left")
        self.entry.pack(side="left")

        entryframe.grid(pady=10)

        b1 = tk.Button(buttonframe, width=30, text="Create New Profile",command=self.create_profile)
        b2 = tk.Button(buttonframe, width=30, text="Load Profile", command=self.load_profile)
        b3 = tk.Button(buttonframe, width=30, text="Quit", command=parent.quit)
        
        b1.grid()
        b2.grid()
        b3.grid()

        buttonframe.place(in_=self, anchor="c", relx=.5, rely=.5)

    def create_profile(self):
        global model, profileName, raw_x, raw_y
        name = self.entry.get()

        if not name:
            # if no name is given
            messagebox.showerror("Profile Error","Invalid name.")
            return

        # Check if the files exist
        try:
            with open(MODEL_FILE.format(name),'rb') as modelFile:
                pass
            with open(RAW_FILE.format(name),'rb') as rawFile:
                pass
            messagebox.showerror("Profile Error","Profile already exists.") 
        except FileNotFoundError:
            # no previous profile data
            profileName = name
            self.parent.set_profile(name)
            raw_x, raw_y = [],[]
            model = 0 
            messagebox.showinfo("Create Profile",
                                "Profile successfully created. \
                                \nNote that empty profiles are authomatically deleted.")
            self.entry.delete(0,'end')
            self.parent.goto_main()

    def load_profile(self):
        global model, profileName, raw_x, raw_y
        name = self.entry.get()
        try:
            with open(MODEL_FILE.format(name),'rb') as modelFile:
                model = pickle.load(modelFile)
            with open(RAW_FILE.format(name),'rb') as rawFile:
                x_vect, y_vect = pickle.load(rawFile)
            profileName = name
            self.parent.set_profile(name)
            messagebox.showinfo("Profile Load","Profile successfully loaded.")
            self.entry.delete(0,'end')
            self.parent.goto_main()
        except FileNotFoundError:
            messagebox.showerror("Profile Error","Profile not found.")
            

class MainPage(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.lastState = True

        buttonframe = tk.Frame(self)

        self.b1 = tk.Button(buttonframe, width=30, text="Record Data", command=parent.goto_record)
        self.b2 = tk.Button(buttonframe, width=30, text="Start Classification", command=parent.goto_predict)
        b3 = tk.Button(buttonframe, width=30, text="Back", command=parent.goto_profile)

        self.b1.pack()
        self.b2.pack()
        b3.pack()

        buttonframe.place(in_=self, anchor="c", relx=.5, rely=.5)

    def set_buttons(self,turnOn=True):
        # this bit of lastState code isn't strictly needed
        # it just makes the GUI look slightly better
        if self.lastState == turnOn:
            return
        else:
            self.lastState = turnOn

        if turnOn:
            self.b1['state'] = 'normal'
            self.b2['state'] = 'normal'
        else:
            self.b1['state'] = 'disabled'
            self.b2['state'] = 'disabled'

class RecordPage(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent

        progressframe = tk.Frame(self)

        self.pBar = ttk.Progressbar(progressframe,orient='horizontal',length=200,
                                    mode='determinate',maximum=STACK_SIZE/10 - 6)
        # the - 6 in maximum is just to get the bar to look right
        # it has no effect on actual timing, which is controlled by start_record
        self.pBar.grid()

        selectframe = tk.Frame(progressframe)
        selectLabel = tk.Label(selectframe, text="Recording for input")
        self.selection = tk.StringVar(selectframe)
        self.selection.set(LABEL_NAME[0])
        self.selector = tk.OptionMenu(selectframe, self.selection, *LABEL_NAME)

        selectframe.grid(pady=10)

        selectLabel.pack(side="left")
        self.selector.pack(side="left")

        self.b1 = tk.Button(progressframe, width=30, text="Record",command=self.start_record)
        self.b2 = tk.Button(progressframe, width=30, text="Generate and Save Model",command=self.generate_save)
        self.b3 = tk.Button(progressframe, width=30, text="Clear Data",command=self.clear_data)
        self.b4 = tk.Button(progressframe, width=30, text="Back",command=parent.goto_main)

        self.b1.grid()
        self.b2.grid()
        self.b3.grid()
        self.b4.grid()

        progressframe.place(in_=self, anchor="c", relx=.5, rely=.5)

    def start_record(self):
        self.pBar.start()
        self.selector['state'] = 'disabled'
        self.b1['state'] = 'disabled'
        self.b2['state'] = 'disabled'
        self.b3['state'] = 'disabled'
        self.b4['state'] = 'disabled'
        self.after(STACK_SIZE*5,self.stop_record)

    def stop_record(self):
        global freqSpec, raw_x, raw_y
        self.pBar.stop()

        update_data()
        raw_x.append(freqSpec)
        raw_y.append(LABEL_NAME.index(self.selection.get() ) )
        print(len(freqSpec[0]))

        self.selector['state'] = 'normal'
        self.b1['state'] = 'normal'
        self.b2['state'] = 'normal'
        self.b3['state'] = 'normal'
        self.b4['state'] = 'normal'

    def generate_save(self):
        global profileName, raw_x, raw_y, model

        if not profileName:
            # if no profile is loaded
            messagebox.showerror("Profile Error","No profile loaded.")
            return

        print("Training model...")
        model = train_model(raw_x,raw_y)
        print("Saving...")
        # Opening files as wb+ will create files that don't exist
        with open(MODEL_FILE.format(profileName),'wb+') as modelFile:
            pickle.dump(model, modelFile)
        with open(RAW_FILE.format(profileName),'wb+') as rawFile:
            pickle.dump((raw_x,raw_y), rawFile)

        messagebox.showinfo("Generate and Save",
                            "Machine Learning Model has been successfully generated and saved.")

    def clear_data(self):
        global raw_x, raw_y
        if messagebox.askokcancel("Clear Dataset",
                                  "Clear all data in this profile?"):
            raw_x, raw_y = [],[]
            messagebox.showinfo("Clear Dataset","Profile cleared.")

class PredictPage(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.active = False
        self.mazeActive = False
        self.lastPred = 0

        SIZE = (400,400)
        DIM = (5,5)
        LAYOUT = [[0,0,0,0,2],
                  [0,1,0,0,0],
                  [0,0,0,1,1],
                  [0,1,0,0,0],
                  [0,1,0,0,0]]
        self.START = (0,4)
        self.SIZE = SIZE
        self.DIM = DIM
        self.LAYOUT = LAYOUT

        self.cur_x, self.cur_y = self.START
        self.dir = 0

        mazeframe = tk.Frame(self)

        mazeCanvas = tk.Canvas(mazeframe,width=SIZE[0],height=SIZE[1],bd=0)
        # borders
        mazeCanvas.create_line(1,1,SIZE[0],1)
        mazeCanvas.create_line(1,1,1,SIZE[1])
        mazeCanvas.create_line(SIZE[0],1,SIZE[0],SIZE[1])
        mazeCanvas.create_line(1,SIZE[1],SIZE[0],SIZE[1])
        x,y = 0,0
        # verticals
        for i in range(0,SIZE[0]):
            x += SIZE[0]/DIM[0]
            mazeCanvas.create_line(x,1,x,SIZE[1])
        # horizontals
        for i in range(0,SIZE[0]):
            y += SIZE[1]/DIM[1]
            mazeCanvas.create_line(1,y,SIZE[0],y)
        # walls, end zone
        x,y = SIZE[0]/DIM[0],SIZE[1]/DIM[1]
        for j in range(0,DIM[1]):
            for i in range(0,DIM[0]):
                if LAYOUT[j][i] == 1:
                    mazeCanvas.create_rectangle(i*x,j*y,(i+1)*x,(j+1)*y,fill='black')
                elif LAYOUT[j][i] == 2:
                    mazeCanvas.create_rectangle(i*x,j*y+1,(i+1)*x,(j+1)*y,fill='green')
        # bot "sprite"
        self.botBody = mazeCanvas.create_oval((self.cur_x+0.5)*x-30,(self.cur_y+0.5)*y-30,
                                               (self.cur_x+0.5)*x+30,(self.cur_y+0.5)*y+30,
                                               fill='#424242')
        self.botArrow = mazeCanvas.create_polygon((self.cur_x+0.5)*x,(self.cur_y+0.5)*y-20,
                                                  (self.cur_x+0.5)*x+15,(self.cur_y+0.5)*y+15,
                                                  (self.cur_x+0.5)*x-15,(self.cur_y+0.5)*y+15,
                                                  fill='red')

        mazeCanvas.pack()
        self.mazeCanvas = mazeCanvas

        statusframe = tk.Frame(mazeframe)
        statusframe.pack(side="bottom", fill="x")

        self.predictLabel = tk.Label(statusframe, text="Input: -")
        self.b1 = tk.Button(statusframe, text="Start",width=7,command=self.toggle_maze)
        b2 = tk.Button(statusframe, text="Reset",command=self.reset_maze)
        b3 = tk.Button(statusframe, text="Back",command=self.exit)

        self.predictLabel.pack(side='left')
        b3.pack(side='right')
        b2.pack(side='right')
        self.b1.pack(side='right')

        mazeframe.place(in_=self, anchor="c", relx=.5, rely=.5)

    def update_pred(self):
        global model, freqSpec

        if not self.active:
            # IT'S TIME TO STOP
            return

        update_data()
        pred = predict(model,freqSpec)
        self.predictLabel['text'] = "Input: {}".format(LABEL_NAME[pred])

        if (self.mazeActive and self.lastPred != pred):
            self.lastPred = pred
            if (pred != 0):
                self.move_bot(pred)

        self.after(UPDATE_INT*5,self.update_pred)

    def toggle_maze(self):
        if self.mazeActive:
            self.mazeActive = False
            self.b1['text'] = "Start"
        else:
            self.mazeActive = True
            self.b1['text'] = "Stop"

    def reset_maze(self):
        self.mazeActive = False
        self.b1['text'] = "Start"
        self.cur_x,self.cur_y = self.START
        self.dir = 0
        self.draw_bot(self.cur_x,self.cur_y,self.dir)

    def move_bot(self, action):
        if action == 1:
            if (self.dir == 0 and self.cur_y > 0 and \
                self.LAYOUT[self.cur_y-1][self.cur_x] != 1):
                self.cur_y -= 1 # go up
            elif (self.dir == 1 and self.cur_x < self.DIM[0]-1 and \
                self.LAYOUT[self.cur_y][self.cur_x+1] != 1):
                self.cur_x += 1 # go right
            elif (self.dir == 2 and self.cur_y < self.DIM[1]-1 and \
                self.LAYOUT[self.cur_y+1][self.cur_x] != 1):
                self.cur_y += 1 # go down
            elif (self.dir == 3 and self.cur_x > 0 and \
                self.LAYOUT[self.cur_y][self.cur_x-1] != 1):
                self.cur_x -= 1 # go left
        elif action == 2: # turn left
            if self.dir == 0:
                self.dir = 3
            else:
                self.dir -= 1
        elif action == 3: # turn right
            if self.dir == 3:
                self.dir = 0
            else:
                self.dir += 1
        # update the sprite
        self.draw_bot(self.cur_x,self.cur_y,self.dir)


    def draw_bot(self,x,y,dir):
        # MANUAL DRAWING IS HELL
        xGrid,yGrid = self.SIZE[0]/self.DIM[0],self.SIZE[1]/self.DIM[1]

        self.mazeCanvas.coords(self.botBody,
                               (x+0.5)*xGrid-30,(y+0.5)*yGrid-30,
                               (x+0.5)*xGrid+30,(y+0.5)*yGrid+30)
        if dir == 0: # UP
            self.mazeCanvas.coords(self.botArrow,
                                   (x+0.5)*xGrid,(y+0.5)*yGrid-20,
                                   (x+0.5)*xGrid+15,(y+0.5)*yGrid+15,
                                   (x+0.5)*xGrid-15,(y+0.5)*yGrid+15)
        if dir == 1: # RIGHT
            self.mazeCanvas.coords(self.botArrow,
                                   (x+0.5)*xGrid+20,(y+0.5)*yGrid,
                                   (x+0.5)*xGrid-15,(y+0.5)*yGrid+15,
                                   (x+0.5)*xGrid-15,(y+0.5)*yGrid-15)
        if dir == 2: # DOWN
            self.mazeCanvas.coords(self.botArrow,
                                   (x+0.5)*xGrid,(y+0.5)*yGrid+20,
                                   (x+0.5)*xGrid+15,(y+0.5)*yGrid-15,
                                   (x+0.5)*xGrid-15,(y+0.5)*yGrid-15)
        if dir == 3: # LEFT
            self.mazeCanvas.coords(self.botArrow,
                                   (x+0.5)*xGrid-20,(y+0.5)*yGrid,
                                   (x+0.5)*xGrid+15,(y+0.5)*yGrid+15,
                                   (x+0.5)*xGrid+15,(y+0.5)*yGrid-15)

    def exit(self):
        self.active = False
        self.reset_maze()
        self.parent.goto_main()

class MainApp(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        # pages
        self.profile = ProfilePage(self)
        self.main = MainPage(self)
        self.record = RecordPage(self)
        self.predict = PredictPage(self)

        profileframe = tk.Frame(self)
        container = tk.Frame(self)
        statusframe = tk.Frame(self)

        profileframe.pack(side="top", fill="x", expand=False)
        container.pack(side="top", fill="both", expand=True)
        statusframe.pack(side="bottom", fill="x")

        # pages
        self.profile.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        self.main.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        self.record.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        self.predict.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        # top bar
        titleLabel = tk.Label(profileframe, text="Current Profile: ")
        self.profileLabel = tk.Label(profileframe, text="-")

        titleLabel.pack(side="left")
        self.profileLabel.pack(side="left")

        # bottom bar
        self.statusCanvas = tk.Canvas(statusframe,width=10,height=10)
        self.statusLight = self.statusCanvas.create_oval(2,2,10,10,fill='red')
        self.statusLabel = tk.Label(statusframe, text="Not Connected")

        self.statusCanvas.pack(side="left")
        self.statusLabel.pack(side="left")
        
        # debug buttons destroy when done
        b1 = tk.Button(profileframe, text="Profile", command=self.profile.lift)
        b2 = tk.Button(profileframe, text="Main", command=self.main.lift)
        b3 = tk.Button(profileframe, text="Record", command=self.record.lift)
        b4 = tk.Button(profileframe, text="Predict", command=self.predict.lift)
        b4.pack(side="right")
        b3.pack(side="right")
        b2.pack(side="right")
        b1.pack(side="right")
        # debug buttons end here

        # start page
        self.profile.lift()
        self.update_conn()

    def goto_main(self):
        self.main.lift()

    def goto_profile(self):
        self.profile.lift()

    def goto_record(self):
        self.record.lift()

    def goto_predict(self):
        self.predict.active = True
        self.predict.lift()
        self.predict.update_pred()

    def quit(self):
        global root
        root.destroy()

    def set_profile(self, name):
        self.profileLabel['text'] = name

    def update_conn(self):
        global connected
        if connected:
            self.statusCanvas.itemconfig(self.statusLight, fill='green')
            self.statusLabel['text'] = "Connected"
            self.main.set_buttons(True)
        else:
            self.statusCanvas.itemconfig(self.statusLight, fill='red')
            self.statusLabel['text'] = "Not Connected"
            self.main.set_buttons(False)
        self.after(1000,self.update_conn)

if __name__ == "__main__":
    root = tk.Tk()
    GUI = MainApp(root)
    GUI.pack(side="top", fill="both", expand=True)
    root.title("BCI Project - Version 1.0")
    root.wm_geometry("500x500")
    GanglionThread().start()
    root.mainloop()
    # Only runs when exiting
    exit = True