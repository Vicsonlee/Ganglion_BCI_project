import tkinter as tk

class ProfilePage(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        buttonframe = tk.Frame(self)

        b1 = tk.Button(buttonframe, width=30, text="Create New Profile")
        b2 = tk.Button(buttonframe, width=30, text="Load Profile")
        b3 = tk.Button(buttonframe, width=30, text="Quit", command=parent.quit)
        b4 = tk.Button(buttonframe, width=30, text="goto Main", command=parent.goto_main)

        b1.pack()
        b2.pack()
        b3.pack()
        b4.pack()

        buttonframe.place(in_=self, anchor="c", relx=.5, rely=.5)


class MainPage(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        buttonframe = tk.Frame(self)

        b1 = tk.Button(buttonframe, width=30, text="Record Data", command=parent.goto_record)
        b2 = tk.Button(buttonframe, width=30, text="Start Classification", command=parent.goto_predict)
        b3 = tk.Button(buttonframe, width=30, text="Back", command=parent.goto_profile)

        b1.pack()
        b2.pack()
        b3.pack()

        buttonframe.place(in_=self, anchor="c", relx=.5, rely=.5)

class RecordPage(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Record Page WIP :D")
        label.pack(side="top", fill="both", expand=True)

class PredictPage(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Predict Page WIP :D")
        label.pack(side="top", fill="both", expand=True)

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

        profileframe.pack(side="top", fill="x", expand=False)
        container.pack(side="top", fill="both", expand=True)

        self.profile.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        self.main.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        self.record.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        self.predict.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        titleLabel = tk.Label(profileframe, text="Current Profile: ")
        profileLabel = tk.Label(profileframe, text="GUI_TEST")

        titleLabel.pack(side="left")
        profileLabel.pack(side="left")

        # debug buttons destroy when done
        b1 = tk.Button(profileframe, text="Profile", command=self.profile.lift)
        b2 = tk.Button(profileframe, text="Main", command=self.main.lift)
        b1.pack(side="right")
        b2.pack(side="right")
        # debug buttons end here

        # start page
        self.profile.lift()

    def goto_main(self):
        self.main.lift()

    def goto_profile(self):
        self.profile.lift()

    def goto_record(self):
        self.record.lift()

    def goto_predict(self):
        global root
        self.predict.lift()
        root.wm_geometry("600x600")

    def quit(self):
        global root
        root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    GUI = MainApp(root)
    GUI.pack(side="top", fill="both", expand=True)
    root.title("BCI Project - Version 0.2")
    root.wm_geometry("400x400")
    root.mainloop()