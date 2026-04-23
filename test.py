with open("./dataset/letters.txt", 'r') as f:
    letters = [w.replace("\n", "") for w in f.readlines()]

with open("./dataset/bad_words.txt", 'r') as f:
    bad_words = [w.replace("\n", "") for w in f.readlines()]

with open("./dataset/other_words.txt", 'r') as f:
    words = [w.replace("\n", "") for w in f.readlines()]

with open("output1.txt", "w") as f:
    for arr in (letters, bad_words, words):
        s = "["
        for w in arr:
            s += f"'{w}', "
        s += ']\n'
        f.write(s)
