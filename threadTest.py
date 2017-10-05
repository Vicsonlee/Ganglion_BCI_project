import threading
import time

class Subthread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        global current
        n = 0
        while (n<500):
            time.sleep(1)
            n += 1
            current = n

current = 0
Subthread().start()
while True:
    input("Press Enter for instant n!")
    print(current)