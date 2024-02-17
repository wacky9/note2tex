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

def writeFileHeader(latexFile, titleName, authorName):
    print("\\documentclass{article}", file=latexFile)
    print("\\usepackage[utf8]{inputenc}", file=latexFile)
    print("\\usepackage[english]{babel}", file=latexFile)
    print("\\usepackage{enumitem}", file=latexFile)
    print("\\usepackage{amsmath}", file=latexFile)
    print("\\usepackage{graphicx}", file=latexFile)
    print("\\usepackage[]{amsthm}", file=latexFile)
    print("\\usepackage[]{amssymb}", file=latexFile)
    print("\\theoremstyle{remark}", file=latexFile)
    print("\\title{" + titleName + "}", file=latexFile)
    print("\\author{" + authorName + "}", file=latexFile)
    print("\\date\\today", file=latexFile)
    print("\\begin{document}", file=latexFile)
    print("\\maketitle", file=latexFile)

def writeFileFooter(latexFile):
    print("\\end{document}", file=latexFile)

def writeToken(latexFile, token):
    pass

def writeNewLine(latexFile):
    print("\\\\", file=latexFile)

if __name__ == '__main__':
    latexFile = open("latex.tex", "w")
    writeFileHeader(latexFile, "Test", "Me")
    writeNewLine(latexFile)
    writeFileFooter(latexFile)
    
