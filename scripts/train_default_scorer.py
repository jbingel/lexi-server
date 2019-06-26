from lexi.core.simplification.lexical import LexiScorer
from lexi.core.featurize.featurizers import LexicalFeaturizer
from lexi.core.featurize.functions import *
from lexi.config import FEATURIZER_PATH_TEMPLATE
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics.ranking import precision_recall_curve
from scipy.stats.stats import spearmanr

lf = LexicalFeaturizer()
lf.add_feature_function(WordLength())
lf.add_feature_function(SentenceLength())
lf.add_feature_function(IsAlpha())
lf.add_feature_function(IsLower())
lf.add_feature_function(IsNumerical())

items, y = [], []

for line in open("res/danish_ls_data.tsv"):
    line = line.strip().split("\t")
    if line:
        s = line[0]
        so = int(line[1])
        eo = int(line[2])
        w = s[so:eo]
        items.append((w, s, so, eo))
        y.append(int(line[-1]))

x = lf.featurize(items, fit=True, scale_features=True)
lf.save(FEATURIZER_PATH_TEMPLATE.format("default"))

# x = [lf.featurize(*item)[0] for item in items]
# x = lf.featurize_batch(items)

# y = np.array([0 if x[i][-1] < .2 else 1 for i in range(len(x))])
y = np.array(y).reshape([-1, 1])
# y = np.ones(len(y))

for i in range(10):
    print(x[i], y[i])

print("\n\nLS")

ls = LexiScorer("default", lf, [])
# print(list(ls.model.parameters()))
ls.train_model(x, y, epochs=1000, patience=10)
# print(list(ls.model.parameters()))
p = np.array(ls.predict(x))
print(p.mean())
print(spearmanr(y, p))
ls.save()

for i in range(10):
    print(x[i], y[i], p[i])

# sys.exit(0)
#
#
# print("\n\nDT")
#
# dt = DecisionTreeClassifier()
# dt.fit(x, y)
# p = dt.predict(x)
# print(accuracy_score(y, p))

print("\n\nMLP")

mlp = MLPRegressor(max_iter=1000, warm_start=True, hidden_layer_sizes=[10])
mlp.fit(x, y.reshape(-1))
p = mlp.predict(x)


print(p.mean())
# print(accuracy_score(y, p))

print(spearmanr(y, p))
for i in range(10):
    print(x[i], y[i], p[i])
