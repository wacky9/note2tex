from tkinter import Canvas, Tk, Button
from PIL import Image, ImageDraw
import datetime
from old_code.recognition import *

OUTPUT_FOLDER = 'output'

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")

        self.labels = sorted(get_label_list())
        self.model = init_model()

        # Scale factor for canvas size to image size
        scale_factor = 10

        # Canvas for drawing
        canvas_width = 32 * scale_factor
        canvas_height = 32 * scale_factor
        self.canvas = Canvas(root, width=canvas_width, height=canvas_height, bg='white')
        self.canvas.pack()

        # Button to save drawing
        save_button = Button(root, text="Classify", command=self.classify)
        save_button.pack()

        # Button to clear drawing
        clear_button = Button(root, text="Clear", command=self.clear_drawing)
        clear_button.pack()

        # Initialize drawing
        self.image = Image.new("L", (32, 32), "white")  # "L" mode for black and white
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        # Scale the coordinates for drawing on the larger canvas
        x, y = event.x, event.y
        scaled_x1, scaled_y1 = x - 5, y - 5  # Adjust the scale factor as needed
        scaled_x2, scaled_y2 = x + 5, y + 5  # Adjust the scale factor as needed

        # Draw on the larger canvas (rectangle instead of oval)
        self.canvas.create_rectangle(scaled_x1, scaled_y1, scaled_x2, scaled_y2, fill="black", width=1)

        # Draw on the smaller image (rectangle instead of line)
        small_x1, small_y1 = scaled_x1 // 10, scaled_y1 // 10
        small_x2, small_y2 = scaled_x2 // 10, scaled_y2 // 10
        self.draw.rectangle([small_x1, small_y1, small_x2, small_y2], fill="black")

    def classify(self):
        filename = str(OUTPUT_FOLDER+'/'+datetime.datetime.now().strftime("_%H%M%S")) + ".png"
        self.image.save(filename)

        frame = io.imread(filename)
        predicted_label, confidence = predict_label(frame, self.model, self.labels)
        print(predicted_label, ',', confidence)


    def clear_drawing(self):
        # Clear both the canvas and the drawn image
        self.canvas.delete("all")
        self.image = Image.new("L", (32, 32), "white")
        self.draw = ImageDraw.Draw(self.image)

if __name__ == "__main__":
    root = Tk()
    app = DrawingApp(root)
    root.mainloop()
