import numpy as np
import timeit

to_send = np.arange(0,100)
reciever = []

def loop1(reciever, to_send):
    for i in to_send:
        reciever.append(i)

def loop2(reciever, to_send):
    reciever = list(map(reciever.append, to_send))

def clear():
    reciever = []

def set_work(work):
    to_send = np.arrange(0, work)

setup = '''
import numpy as np

to_send = np.arange(0,100)
reciever = []

def loop1(reciever, to_send):
    for i in to_send:
        reciever.append(i)

def loop2(reciever, to_send):
    reciever = list(map(reciever.append, to_send))

def clear():
    reciever = []

def set_work(work):
    to_send = np.arrange(0, work)
'''
