from lexi.core.simplification.lexical import LexiScorer
from lexi.core.featurize.featurizers import LexicalFeaturizer

lf = LexicalFeaturizer()


items, y = [], []

for line in open("res/danish_ls_data.tsv"):
    line = line.strip().split("\t")
    if line:
        items.append((line[0], int(line[1]), int(line[2])))
        y.append(int(line[-1]))

print(items[:4])
lf.fit(items)
lf.save("default_featurizer.json")

x = [lf.featurize(*item) for item in items]

print(x)
print(y)

ls = LexiScorer("default", lf, [20, 20])

ls.train_model(x, y)
ls.save()
