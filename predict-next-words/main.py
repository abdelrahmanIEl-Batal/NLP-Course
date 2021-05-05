from nltk.corpus import brown
from collections import defaultdict


# to download data into machine, comment if already installed
# nltk.download()


def preprocess(words):
    filter = ["?", ".", "~", "!", ","]
    # remove special characters
    words = [word for word in words if word not in filter]
    for i in range(len(words)):
        words[i] = words[i].lower()
    return words


def initializeCount(words):
    countTwo = defaultdict(dict)
    countThree = defaultdict(lambda: defaultdict(dict))
    probabilityThree = defaultdict(lambda: defaultdict(dict))

    for i in range(len(words) - 2):
        # last two words in countTwo not initialised, and will be increased by 1 kda kda fa n3mlha hna w5las
        if i == len(words) - 2:
            countTwo[words[i + 1]][words[i + 2]] = 1
        countTwo[words[i]][words[i + 1]] = 0
        countThree[words[i]][words[i + 1]][words[i + 2]] = 0

    # calculate count
    for i in range(len(words) - 2):
        countTwo[words[i]][words[i + 1]] += 1
        countThree[words[i]][words[i + 1]][words[i + 2]] += 1

    # p(z| x, y) = c(x,y,z) / c(x,y)
    for i in range(len(words) - 2):  # add backlash and try later
        probabilityThree[words[i]][words[i + 1]][words[i + 2]] = countThree[words[i]][words[i + 1]][words[i + 2]] / \
                                                                 countTwo[words[i]][words[i + 1]]

    return countThree, probabilityThree


def predictWord(sentence, countThree, probabilityThree):
    # should be 2 words, input len is 2
    input = sentence.split()
    res = defaultdict(dict)
    # print(len(probabilityThree))
    for thirdWord in countThree[input[0]][input[1]]:
        res[thirdWord] = probabilityThree[input[0]][input[1]][thirdWord]

    sortedProbabilities = [(key, res[key]) for key in sorted(res, key=res.get, reverse=False)]

    predictedWords = []

    for key, value in sortedProbabilities:
        predictedWords.append(key)
    return predictedWords[0:5]


# 240k word for the following categories
words = brown.words(categories=['news'])
words = preprocess(words)

countThree, probabilityThree = initializeCount(words)

input = "grand jury"

res = predictWord(input, countThree, probabilityThree)
for w in res:
    print(w)
