from sklearn import ensemble
import pickle
import numpy as np
import copy

WORD2VEC_EMBEDDINGS = '/Users/marcotcr/phd/datasets/word2vec/our_dataset_embeddings.pickle'
class EmbeddingForest():
  def __init__(self, vectorizer,
  embedding_path=WORD2VEC_EMBEDDINGS,
  inverse_vocabulary = None):
    if inverse_vocabulary is not None:
      self.inverse_vocabulary = inverse_vocabulary
    else:
      terms = np.array(list(vectorizer.vocabulary_.keys()))
      indices = np.array(list(vectorizer.vocabulary_.values()))
      self.inverse_vocabulary = terms[np.argsort(indices)]
    self.embeddings = pickle.load(open(embedding_path))
    self.classifier = ensemble.RandomForestClassifier(n_estimators=1000, random_state=1, class_weight='balanced_subsample')
  def transform_example(self, X):
    # X is a sparse vector or sparse matrix
    ret = []
    for v in X:
      words = [self.inverse_vocabulary[x] for x in v.nonzero()[1]]
      new = np.array([self.embeddings[x] for x in words if x in self.embeddings])
      if new.shape[0] == 0:
        new = np.zeros((1,300))
      ret.append(np.mean(new, axis=0))
    ret = np.array(ret)
    return ret
  def fit(self,X,Y):
    emb_x = self.transform_example(X)
    self.classifier.fit(emb_x, Y)
  def predict_proba(self, v):
    return self.classifier.predict_proba(self.transform_example(v))
  def predict(self, v):
    return self.classifier.predict(self.transform_example(v))
  def get_params(self, deep=False):
    #params = self.classifier.get_params(deep)
    params = {}
    params['vectorizer'] = ''
    params['inverse_vocabulary'] = copy.deepcopy(self.inverse_vocabulary)
    return params
    
