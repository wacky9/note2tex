# hack-ai-2024
Hack AI 2024 @ Ohio State. Henry, Ram, Winston &amp; Toby.

Le'Texifier is a streamlined handwritten notes to LaTeX converter designed to save scientists, mathematicians, and any LaTeX user countless hours.
The program reads in your handwritten math notes taken on an ipad and outputs LaTeX code with a corresponding generated pdf. This is accomplished
by performing image segmentation on symbols by using canny edge detection and by extracting bounding boxes. This is then fed into a neural network
our team trained using a combination of our own data, mnist data, and synthetic data we generated. This is then finally passed into a python script we wrote
to map these neural network predictions into a LaTeX file and a generated pdf.
