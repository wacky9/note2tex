import csv
import os

tokenMapping = {
  "\plus": "+",
  "\minus": "-",
  "\\times": "\\cdot",
  "\divide": "/",

  "\equals": "=",
  "\leq": "\\leq",
  "\geq": "\\geq",
  "\\notEqual": "\\neq",

  "\<": "<",
  "\>": ">",
  "\|": "|",


  "\\startsqrt": "\sqrt{",
  "\\endsqrt": "}",
  "\startexp": "^{",
  "\endexp": "}",
  "\startSubscript": "_{",
  "\endSubscript": "}",
  "\startFrac": "\\frac{",
  "\endNumerator": "}{",
  "\endFrac": "}",

  "\comma": ",",

  "\\rightArrow": "\\Rightarrow ",
  "\leftArrow": "\\Leftarrow ",
  "\doubleArrow": "\\Leftrightarrow ",

  "\subset": "\\subset ",

  "\\bbR": "\\mathbb{R} ",
  "\\bbZ": "\\mathbb{Z} ",
  "\\bbN": "\\mathbb{N} ",
  "\\bbQ": "\\mathbb{Q} ",
  "\\bbC": "\\mathbb{C} ",

  "\pi": "\\pi ",
  "\epsilon": "\\epsilon ",
  "\delta": "\\delta ",

  "\\forall": "\\forall ",
  "\exists": "\\exists ",


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
    print("\\noindent", file=latexFile)
    latexFile.write("$")

def writeFileFooter(latexFile):
    print("$\n\n\\end{document}", file=latexFile)

def writeToken(latexFile, token, inMath):
    if token[0] == "\\":
        if not inMath:
            latexFile.write("}")
            inMath = True
        latexFile.write(tokenMapping[token])
    else:
        if inMath:
            latexFile.write("\\textrm{")
            inMath = False
        latexFile.write(token)
    return inMath

def writeNewLine(latexFile, inMath):
    if not inMath:
        latexFile.write("}")
    latexFile.write("$\\\\\n$")

def generateLatexPdf():
    #Generate the pdf
    execution_string = 'pdflatex  --max-print-line=10000 -synctex=1 -interaction=nonstopmode -file-line-error -recorder  "c:/Users/User/Desktop/hack-ai-2024/latex.tex"'
    execution_string_local = 'pdflatex  --max-print-line=10000 -synctex=1 -interaction=nonstopmode -file-line-error -recorder  "./latex.tex"'
    os.system(execution_string_local)


if __name__ == '__main__':
    
    #Open the generated latex file and the intermediate csv file.
    latexFile = open("latex.tex", "w")
    intermediateFile = open("intermediate.csv", "r")

    #Output the latex header
    writeFileHeader(latexFile, "Test", "Me")

    # Read CSV file
    csvReader = csv.reader(intermediateFile, delimiter=',')
    rows = list(csvReader)  # Store rows in a list
    numRows = len(rows)
    inMath = True

    for i, row in enumerate(rows, start=1):
        for token in row:
            inMath = writeToken(latexFile, token, inMath)
        if i < numRows:
            writeNewLine(latexFile, inMath)
            inMath = True

    #Output the latex footer
    writeFileFooter(latexFile)

    #Generate the latex pdf
    generateLatexPdf()
    
