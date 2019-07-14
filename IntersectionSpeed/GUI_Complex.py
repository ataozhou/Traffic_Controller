import Tkinter as tk
from PIL import ImageTk, Image
import time

class GUI():

    def __init__(self, numRoads, roadLength, grid_space):
        self.grid_space = grid_space
        self.numRoads = numRoads
        self.roadLength = roadLength
        self.width = 1000
        self.height = 1000
        self.EWCars = []
        self.NSCars = []
        self.root = tk.Tk()
        self.simpleCarNS = ImageTk.PhotoImage(Image.open("./RedCar.png").resize(((grid_space/2),(grid_space/2))).rotate(-90))
        self.simpleCarEW = ImageTk.PhotoImage(Image.open("./BlueCar.png").resize(((grid_space/2),(grid_space/2))))
        self.boom = ImageTk.PhotoImage(Image.open("./boom.jpg").resize(((grid_space/2),(grid_space/2))))
        self.c = tk.Canvas(self.root, height=1000, width=1000, bg='white')
        self.c.pack(fill = tk.BOTH, expand= tk.YES)
        self.c.bind('<Configure>', self.create)


    def create(self, event=None):
        w = self.c.winfo_width() # Get current width of canvas
        h = self.c.winfo_height() # Get current height of canvas

        # Creates all vertical lines at intevals of self.grid_space
        for i in range(0, w, self.grid_space):
            self.c.create_line([(i, 0), (i, h)], tag='grid_line')
            label = tk.Label(self.c, text = str(i/50), fg = 'white', bg = 'black')
            #label.pack()
            self.c.create_window(i-(self.grid_space/2), (self.grid_space)/2 , window = label)

        # Creates all horizontal lines at intevals of self.grid_space
        for i in range(0, h, self.grid_space):
            self.c.create_line([(0, i), (w, i)], tag='grid_line')
            label = tk.Label(self.c, text = str(i/50), fg = 'white', bg = 'black')
            #label.pack()
            self.c.create_window((self.grid_space/2), i - (self.grid_space/2) , window = label)

        #label = tk.Label(self.c, text = "intersection", fg = 'white', bg = 'black', font=("Helvetica", 6))
        #label.pack()
        #self.c.create_window(w/2 - (self.grid_space/2), h/2 - (self.grid_space * .9), window = label)
        self.height = h
        self.width = w

    def update(self, move, carIn):
        center_x = (self.width / 2) - (self.grid_space/2)  # Get current width of canvas
        center_y = (self.height / 2) - (self.grid_space/2) # Get current height of canvas
        canvas = self.c

        for i in range(len(NSCars)):
        	canvas.move(NSCars[i], 0, (move[0][i] * self.grid_space))

        for i in range(len(EWCars)):
        	canvas.move(EWCars[i], (-1 * move[1][i] * self.grid_space), 0)

        '''#NS Road
        x = center_x
        y = center_y - self.grid_space
        for bit in state[0]:
            if bit == 1:
                pic = tk.Label(canvas, image = self.simpleCarNS)
                pic.pack()
                canvas.create_window(x, y, window = pic)
            y += self.grid_space


        #EW Road
        x = center_x - self.grid_space
        y = center_y
        for bit in state[1]:
            if bit == 1:
                pic = tk.Label(canvas,image = self.simpleCarEW)
                pic.pack()
                canvas.create_window(x, y, window = pic)
            x += self.grid_space

        if ((state[0][1] + state[1][1]) == 2): 
            label = tk.Label(canvas, image = self.boom)
            label.pack()
            canvas.create_window(center_x, center_y, window = label)

        elif((state[0][1]) == 1):
            pic = tk.Label(canvas, image = self.simpleCarNS)
            pic.pack()
            canvas.create_window(center_x, center_y, window = pic)
        elif((state[1][1]) == 1):
            pic = tk.Label(canvas, image = self.simpleCarEW)
            pic.pack()
            canvas.create_window(center_x, center_y, window = pic)'''


        for i in range(EWCars):
        	if canvas.coords(EWCars[i])[0] <= (center_x - (self.grid_space)):
        		EWCars.pop(i)

        for i in range(NSCars):
        	if canvas.coords(NSCars[i])[1] <= (center_y - (self.grid_space)):
        		NSCars.pop(i)


        #places car in EWRoad
        for car in carIn[0]:
        	posX = center_x + ((self.roadLength + self.numRoads - 1) * self.grid_space)
        	posY = center_y + (car * self.grid_space)
        	pic = tk.Label(canvas, image = self.simpleCarNS)
        	pic.pack()
        	canvas.create_window(posX, posY, window = pic)
        	self.EWCars.append(pic)


        #places car in NSRoad
        for car in carIn[1]:
        	posX = center_x + (car * self.grid_space)
        	posY = center_y + ((self.roadLength + self.numRoads - 1) * self.grid_space)
        	pic = tk.Label(canvas, image = self.simpleCarNS)
        	pic.pack()
        	canvas.create_window(posX, posY, window = pic)
        	self.NSCars.append(pic)

        self.c.update()
        self.c.update_idletasks()
        #time.sleep(2)


    def destroy():
        self.root.quit()
        self.root.destroy()
