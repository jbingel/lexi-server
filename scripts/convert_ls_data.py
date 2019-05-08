import sys


def create_modified_sent(tokens, idx, w, lbl):

    tokens_new = tokens
    tokens[idx] = w
    sent = " ".join(tokens_new)
    startOffset = 0
    for i in range(idx):
        startOffset += len(tokens_new[i]) + 1
    endOffset = startOffset + len(w)

    return "{}\t{}\t{}\t{}".format(sent, startOffset, endOffset, lbl)


infile = open(sys.argv[1])
outfile = open(sys.argv[2], "w")

for line in infile:
    line = line.strip()
    if line:
        sent, w, idx, alt1, alt2 = line.split("\t")
        toks = sent.split()
        out = create_modified_sent(toks, int(idx), alt1.split(":")[1], 0)
        outfile.write(out+"\n")
        out = create_modified_sent(toks, int(idx), alt2.split(":")[1], 1)
        outfile.write(out + "\n")
