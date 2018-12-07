from graphics import *


def main():
    win = GraphWin('Face', 200, 150) # give title and dimensions

    head = Circle(Point(40,100), 25) # set center and radius
    head.setFill("yellow")
    head.draw(win)
    win.getMouse()
    win.close()


main()