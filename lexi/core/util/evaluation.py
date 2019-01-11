from sklearn.metrics import recall_score, precision_score, f1_score, \
    accuracy_score


def evaluate(pred, gold):
    r = recall_score(gold, pred, average="micro", labels=[1, 2])
    p = precision_score(gold, pred, average="micro", labels=[1, 2])
    f1 = f1_score(gold, pred, average="micro", labels=[1, 2])
    a = accuracy_score(gold, pred)
    return f1, r, p, a