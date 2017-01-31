import argparse


def tokenize(text, dictionary, output):
    word2id = {}
    with open(dictionary) as dic:
        for line in dic:
            i, word = line.split()
            word2id[word] = i

    tok = open(output, 'w')
    with open(text) as f:
        for line in f:
            tok.write(" ".join([word2id[w] for w in line.split()])+"\n")

    tok.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenizer')
    parser.add_argument('-d', '--dictionary', action="store", type=str)
    parser.add_argument('-t', '--text', action="store", type=str)
    parser.add_argument('-o', '--output', action="store", type=str)
    args = parser.parse_args()

    tokenize(args.text, args.dictionary, args.output)
