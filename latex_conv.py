
tokenMapping = {
  "\plus": "+",
  "\minus": "\\",
  "\\times": "\\cdot",
  "\divide": "/",
  "!": "!",
  "\comma": ",",

  "(": "(",
  ")": ")",

  "\\startsqrt": "\sqrt{",
  "\\endqrt": "}",

  "\startexp": "^{",
  "\endexp": "}"

}

def writeToken(latexFile, token):
    pass

def writeNewLine(latexFile):
    latexFile.write("\\\\")

if __name__ == '__main__':
    latexFile = open("latex.tex", 'w')
    writeNewLine(latexFile)
