class Word:

    def __init__(self, w):
        self.w = w

    def a_simple_feats(self):
        D = dict()
        D["a_length"] = len(self.w)
        return D

    def featurize_by_type(self, feat_types):
        _d = dict()
        type2func = {'simple': self.a_simple_feats,
                     }
        if not feat_types:
            # use all feats if none explicitly
            feat_types = type2func.keys()
        for feat in feat_types:
            _d.update(type2func[feat]())
        return _d
