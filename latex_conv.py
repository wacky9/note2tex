import csv

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
  "\\endsqrt": "}",

  "\startexp": "^{",
  "\endexp": "}",

  "\pi" : "\\pi",
  "\equals" : "="

}

def writeFileHeader(latexFile, titleName, authorName):
    print("\\documentclass{article}", file=latexFile)
    print("\\usepackage[utf8]{inputenc}", file=latexFile)
    print("\\usepackage[english]{babel}", file=latexFile)
    print("\\usepackage{enumitem}", file=latexFile)
    print("\\usepackage{amsmath}", file=latexFile)
    print("\\usepackage{graphicx}", file=latexFile)
    print("\\usepackage[]{amsthm}", file=latexFile)
    print("\\usepackage[]{amssymb}\n", file=latexFile)
    print("\\theoremstyle{remark}\n", file=latexFile)
    print("\\title{" + titleName + "}", file=latexFile)
    print("\\author{" + authorName + "}", file=latexFile)
    print("\\date\\today\n", file=latexFile)
    print("\\begin{document}", file=latexFile)
    print("\\maketitle\n", file=latexFile)
    latexFile.write("$")

def writeFileFooter(latexFile):
    print("$\n\n\\end{document}", file=latexFile)

def writeToken(latexFile, token):
    if token[0] == "\\":
        latexFile.write(tokenMapping[token])
    else:
        latexFile.write(token)

def writeNewLine(latexFile):
    latexFile.write("$\n\\\\$")

if __name__ == '__main__':
    
    #Open the generated latex file and the intermediate csv file.
    latexFile = open("latex.tex", "w")
    intermediateFile = open("intermediate.csv", "r")

    #Output the latex header
    writeFileHeader(latexFile, "Test", "Me")

    csvReader = csv.reader(intermediateFile, delimiter=',')
    for row in csvReader:
        for token in row:
            writeToken(latexFile, token)
        writeNewLine(latexFile)

    #Output the latex footer
    writeFileFooter(latexFile)
    
