import multiprocessing as mp
import time


def target0(conn):
    for i in range(0,10):
        conn.send("data")

def target1(conn):
    while True:
        if conn.poll():
            data = conn.recv()
            print(data)

rx, tx = mp.Pipe()
p0 = mp.Process(target=target0, args=(tx,))
p1 = mp.Process(target=target1, args=(rx,))

if __name__ == "__main__":
    p0.start()
    p1.start()
    while True:
        try:
            pass
        except KeyboardInterrupt:
            print("exiting")
            p0.join()
            p1.join()
            break
