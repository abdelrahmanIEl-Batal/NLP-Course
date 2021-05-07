from nltk.corpus import brown
from collections import defaultdict
import tkinter as tk
import glob


def preprocess(words):
    filter = ["؟", ".", "~", "!", ","]
    # remove special characters
    words = [word for word in words if word not in filter]
    for i in range(len(words)):
        words[i] = words[i].lower()
    return words


def initializeCount(words):
    countTwo = defaultdict(dict)
    countThree = defaultdict(lambda: defaultdict(dict))
    probabilityThree = defaultdict(lambda: defaultdict(dict))

    # will do bigram as well
    countOne = defaultdict(dict)
    probabilityTwo = defaultdict(lambda: defaultdict(dict))

    for i in range(len(words) - 2):
        countTwo[words[i]][words[i + 1]] = 0
        countThree[words[i]][words[i + 1]][words[i + 2]] = 0
        countOne[words[i]] = 0

    # last two words in countTwo not initialised, and will be increased by 1 kda kda fa n3mlha hna w5las
    countOne[words[len(words) - 1]] = 1
    countOne[words[len(words) - 2]] = 1
    countTwo[words[len(words) - 2]][words[len(words) - 1]] = 1
    # calculate count
    for i in range(len(words) - 2):
        countTwo[words[i]][words[i + 1]] += 1
        countThree[words[i]][words[i + 1]][words[i + 2]] += 1
        countOne[words[i]]+= 1
    # p(z| x, y) = c(x,y,z) / c(x,y)
    for i in range(len(words) - 2):
        probabilityThree[words[i]][words[i + 1]][words[i + 2]] = countThree[words[i]][words[i + 1]][words[i + 2]] / \
                                                                 countTwo[words[i]][words[i + 1]]
    # p(y,x) = c(x,y) / c(x)
    for i in range(len(words) - 1):
        probabilityTwo[words[i]][words[i+1]] = countTwo[words[i]][words[i+1]] / countOne[words[i]]
    return countTwo, probabilityTwo, countThree, probabilityThree


def predictWord(sentence, countTwo, probabilityTwo, countThree, probabilityThree):
    input = sentence.split()
    res = defaultdict(dict)
    if len(input) > 2 or len(input) == 0:
        return "No Results"
    if len(input) == 2:
        for thirdWord in countThree[input[0]][input[1]]:
            res[thirdWord] = probabilityThree[input[0]][input[1]][thirdWord]

    if len(input) == 1:
        for secondWord in countTwo[input[0]]:
            res[secondWord] = probabilityTwo[input[0]][secondWord]

    sortedProbabilities = [(key, res[key]) for key in sorted(res, key=res.get, reverse=True)]
    predictedWords = []

    for key, value in sortedProbabilities:
        predictedWords.append(key)
    return predictedWords[0:5]


def updateGUI(sentence, root, countTwo, probabilityTwo, countThree, probabilityThree):
    for label in root.grid_slaves():
        if int(label.grid_info()["row"]) > 5:
            label.grid_forget()
    result = predictWord(sentence.get(), countTwo, probabilityTwo, countThree, probabilityThree)
    if result == "No Results" or len(result) == 0:
        tk.Label(root, text="No Results Found").grid(row=6, column=1)
    else:
        c = 6
        for res in result:
            tk.Label(root, text=res).grid(row=c, column=1)
            c += 1
    return


path = "Sports/*.html"
files = glob.glob(path)
words = []
for file in files:
    f = open(file, encoding="utf8")
    content = f.read()
    words.extend(content.split())

words = preprocess(words)

countTwo, probabilityTwo, countThree, probabilityThree = initializeCount(words)

# GUI setup
root = tk.Tk()
root.geometry('350x200')
label1 = tk.Label(root, text="AutoComplete app", font=("Arial Bold", 10)).grid(row=0, column=0)
myvar = tk.StringVar()
myvar.trace('w', lambda name, index, mode, myvar=myvar: updateGUI(myvar, root, countTwo, probabilityTwo, countThree, probabilityThree))

label2 = tk.Label(root, text="Search").grid(row=5, column=0)
entry1 = tk.Entry(root, width=30, textvariable=myvar).grid(row=5, column=1)

root.mainloop()
