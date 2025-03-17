# Overview

This project is divided into a simple pipeline. Stage One, the user uploads a photograph of a single page of notes they desire to convert into LaTeX. Stage Two, the image is separated into logical lines of text. Stage Three, the text in these lines is segmented and special cases are identified (for example: fractions). Stage Four, individual symbols are extracted from these lines. Stage Five, these symbols are identified and turned into tokens. Stage Six, these tokens are combined to create a well-formatted LaTeX file. 

## Line Segmentation
Currently lines are segemented by assuming the user inputs an image of ruled paper. Those lines are then identified and used to determine the location of lines. 
## Semantic Segmentation

## Symbol Detection
Currently, symbols are detected via use of a Convolution Neural Network, made using PyTorch. This network is trained on a combination of the MNIST dataset and on custom data. The current model is largely a proof-of-concept. Future plans include using a more robust dataset and building a ResNet Model. 