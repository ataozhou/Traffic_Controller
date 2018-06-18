import Tkinter as tk
import time

class GUI():

    def __init__(self, grid_space):
        self.grid_space = grid_space
        self.width = 1000
        self.height = 1000
        self.root = tk.Tk()
        self.c = tk.Canvas(self.root, height=1000, width=1000, bg='white')
        self.c.pack(fill=tk.BOTH, expand=True)
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

        label = tk.Label(self.c, text = "intersection", fg = 'white', bg = 'black', font=("Helvetica", 6))
        #label.pack()
        self.c.create_window(w/2 - (self.grid_space/2), h/2 - (self.grid_space * .9), window = label)
        self.height = h
        self.width = w

    def update(self, state):
        center_x = (self.width / 2) - (self.grid_space/2)  # Get current width of canvas
        center_y = (self.height / 2) - (self.grid_space/2) # Get current height of canvas
        canvas = self.c
        canvas.delete(tk.ALL)
        self.create()

        #NS Road
        x = center_x
        y = center_y - self.grid_space
        for bit in state[0]:
            label = tk.Label(canvas, text = str(bit), fg = 'white', bg = 'black')
            label.pack()
            canvas.create_window(x, y, window = label)
            y += self.grid_space


        #EW Road
        x = center_x - self.grid_space
        y = center_y
        for bit in state[1]:
            label = tk.Label(canvas, text = str(bit), fg = 'white', bg = 'black')
            label.pack()
            canvas.create_window(x, y, window = label)
            x += self.grid_space

        center = state[0][1] + state[1][1]
        label = tk.Label(canvas, text = str(center), fg = 'white', bg = 'black')
        label.pack()
        canvas.create_window(center_x, center_y, window = label)

        self.c.update()
        time.sleep(2)

    def destroy():
        self.root.quit()
        self.root.destroy()
