import ann
import numpy
import tkinter as tk
from tkinter import filedialog
from PIL import ImageGrab, Image, ImageFilter


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master, relief="raised", bd=2)
        self.master = master

        # Init Neural Network with 28*28 input nodes, 200 hidden nodes, 10 output nodes (for 0-9) and a .1 learning rate
        self.neural_network = ann.NeuralNetwork(28 * 28, 200, 10, .1)

        # Create and place widgets
        self.pack()
        self.create_widgets()

        # Bind mouse button to draw on click
        self.canvas.bind("<B1-Motion>", self.paint)

        # Create toplevel menu
        menubar = tk.Menu(master)
        menubar.add_command(label="Clear", command=self.clear_canvas)
        menubar.add_separator()
        menubar.add_command(label="Classify", command=self.classify_drawing)

        # Save, load, and reset dropdown options
        session_menu = tk.Menu(master)
        session_menu.add_command(label="Save Session", command=self.save_session)
        session_menu.add_command(label="Load Session", command=self.load_session)

        # Train or test from MNIST dataset
        dataset_menu = tk.Menu(master)
        dataset_menu.add_command(label="Train from MNIST dataset", command=self.train_from_dataset)
        dataset_menu.add_command(label="Test with MNIST dataset", command=self.test_from_dataset)

        menubar.add_separator()
        menubar.add_cascade(label="Session", menu=session_menu)
        menubar.add_separator()
        menubar.add_cascade(label="Dataset", menu=dataset_menu)
        master.config(menu=menubar)

        seperator = tk.Frame(master, height=2, bg="grey")
        seperator.pack(fill="both", expand=True)

    def save_session(self):
        self.neural_network.save_session(
            filedialog.asksaveasfilename(initialdir="./saved/", title="Select where to save session",
                                         filetypes=(("session", "*.session"), ("all files", "*.*"))))
        self.console_text.set("Session Saved.")

    def load_session(self):
        self.neural_network.load_session(
            filedialog.askopenfilename(initialdir="./saved/", title="Select session file to load from",
                                       filetypes=(("session", "*.session"), ("all files", "*.*"))))
        self.console_text.set("Session Loaded.")

    def create_widgets(self):
        self.canvas_frame = tk.Frame()
        self.canvas_frame.pack(fill="x", expand="true")
        self.canvas = tk.Canvas(self.canvas_frame, bg="white", width=500, heigh=500)
        self.canvas.pack(expand="true")

        self.console_frame = tk.Frame()
        self.console_frame.pack(fill="y", expand="true")
        self.console_text = tk.StringVar()
        self.console = tk.Label(self.console_frame, textvariable=self.console_text)
        self.console_text.set("Press and drag mouse on canvas to draw")
        self.console.pack()
        self.console_text_2 = tk.StringVar()
        self.console_2 = tk.Label(self.console_frame, textvariable=self.console_text_2)
        self.console_text_2.set("")
        self.console_2.pack()

        self.retrain_frame = tk.Frame()
        self.retrain_frame.pack(fill="y", expand="true", pady=15)
        self.retrain_label_text = tk.StringVar()
        self.retrain_label = tk.Label(self.retrain_frame, textvariable=self.retrain_label_text)
        self.retrain_label_text.set("Correct Label?")
        self.retrain_label.pack(side="top")
        self.correct_label_text = tk.StringVar()
        self.correct_label = tk.Entry(self.retrain_frame, textvariable=self.correct_label_text)
        self.correct_label.pack(side="left", expand="true")
        self.correct_label_text.set("")
        self.add_new_label = tk.Button(self.retrain_frame, text="Retrain Neural Network", command=self.retrain)
        self.add_new_label.pack(side="left", expand="true", padx=5)

    def paint(self, event):
        ''' Draws tiny ovals representing pixels on the canvas'''

        # Get posion of mouse clicks
        x1, y1 = (event.x - 3), (event.y - 3)
        x2, y2 = (event.x + 3), (event.y + 3)

        # Display oval on canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill="black")

    def clear_canvas(self):
        ''' Clears all points on the canvas '''
        self.canvas.delete("all")  # Clear Drawing

    def get_pixels(self):
        # Get screenshot of canvas and save as canvas.png
        x = root.winfo_rootx() + self.canvas.winfo_x()
        y = root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        ImageGrab.grab().crop((x, y, x1, y1)).save("canvas.png")

        # Resize image to 28x28 and get pixel data
        im = Image.open("canvas.png").convert('L')
        width = float(im.size[0])
        height = float(im.size[1])
        newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

        if width > height:  # check which dimension is bigger
            # Width is bigger. Width becomes 20 pixels.
            nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
            if (nheight == 0):  # rare case but minimum is 1 pixel
                nheight = 1
                # resize and sharpen
            img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wtop = int(round(((28 - nheight) / 2), 0))  # caculate horizontal pozition
            newImage.paste(img, (4, wtop))  # paste resized image on white canvas
        else:
            # Height is bigger. Heigth becomes 20 pixels.
            nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
            if (nwidth == 0):  # rare case but minimum is 1 pixel
                nwidth = 1
                # resize and sharpen
            img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
            newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

        newImage.save("sample.png")

        tv = list(newImage.getdata())  # get pixel values

        # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]
        return tva

    def classify_drawing(self):
        # Query neural network with image pixes
        query = self.neural_network.query(self.get_pixels())

        # Get the predicted class and the runner up
        flat = query.flatten()
        flat.sort()
        self.console_text.set("Label Chosen: " + str(numpy.where(query == flat[-1])[0][0]))
        self.console_text_2.set("Runner Up: " + str(numpy.where(query == flat[-2])[0][0]))

    def retrain(self):
        inputs = self.get_pixels()
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(10) + 0.01
        # all_values[0] is the target label for this record
        targets[int(self.correct_label_text.get())] = 0.99
        # Adjust the neural network with a new training example
        self.neural_network.train(inputs, targets)

    def train_from_dataset(self):
        # Epochs and degrees to rotate each image by for more thorough training
        epochs = 10
        to_rotate = 10
        # Train using MNIST handwriting dataset
        self.neural_network.train_with_rotated_numbers(epochs=epochs, to_rotate=to_rotate)

    def test_from_dataset(self):
        accuracy = self.neural_network.test_neural_net()
        self.console_text.set("ANN accuracy = " + str(accuracy))

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.master.title("Digit Classifier")
    root.resizable(width=False, height=False)
    root.geometry('{}x{}'.format(550, 625))
    app.mainloop()
