from abc import ABCMeta, abstractmethod
import numpy as np
import scipy as sp
from sklearn import linear_model
import sklearn.metrics.pairwise

###############################
## Random Explainer
###############################

class RandomExplainer:
  def __init__(self):
    pass

  def reset(self):
    pass

  def explain_instance(self,
                       instance_vector,
                       label,
                       classifier,
                       num_features,
                       dataset):
    nonzero = instance_vector.nonzero()[1]
    explanation = np.random.choice(nonzero, num_features)
    return [(x, 1) for x in explanation]

  def explain(self,
              train_vectors,
              train_labels,
              classifier,
              num_features,
              dataset):
    i = np.random.randint(0, train_vectors.shape[0])
    explanation = self.explain_instance(train_vectors[i], None, None,
                                        num_features, dataset)
    return i, explanation

###############################
## Standalone Explainers
###############################

def most_important_word(classifier, v, class_):
  # Returns the word w that moves P(Y) - P(Y|NOT w) the most for class Y.
  max_index = 0
  max_change = -1
  orig = classifier.predict_proba(v)[0][class_]
  for i in v.nonzero()[1]:
    val = v[0,i]
    v[0,i] = 0
    pred = classifier.predict_proba(v)[0][class_]
    change = orig - pred
    if change > max_change:
      max_change = change
      max_index = i
    v[0,i] = val
  if max_change < 0:
    return -1
  return max_index

def explain_greedy(instance_vector,
                   label,
                   classifier,
                   num_features,
                   dataset=None):
  explanation = []
  z = instance_vector.copy()
  while len(explanation) < num_features:
    i = most_important_word(classifier, z, label)
    if i == -1:
      break
    z[0,i] = 0
    explanation.append(i)
  return [(x, 1) for x in explanation]
def most_important_word_martens(predict_fn, v, class_):
  # Returns the word w that moves P(Y) - P(Y|NOT w) the most for class Y.
  max_index = 0
  max_change = -1
  orig = predict_fn(v)[0,class_]
  for i in v.nonzero()[1]:
    val = v[0,i]
    v[0,i] = 0
    pred = predict_fn(v)[0,class_]
    change = orig - pred
    if change > max_change:
      max_change = change
      max_index = i
    v[0,i] = val
  if max_change < 0:
    return -1, max_change
  return max_index, max_change

def explain_greedy_martens(instance_vector,
                   label,
                   predict_fn,
                   num_features,
                   dataset=None):
  if not hasattr(predict_fn, '__call__'):                                                 
      predict_fn = predict_fn.predict_proba
  explanation = []
  z = instance_vector.copy()
  cur_score = predict_fn(instance_vector)[0, label]
  while len(explanation) < num_features:
    i, change = most_important_word_martens(predict_fn, z, label)
    cur_score -= change
    if i == -1:
      break
    explanation.append(i)
    if cur_score < .5:
      break
    z[0,i] = 0
  return [(x, 1) for x in explanation]

def data_labels_distances_mapping_text(x, classifier_fn, num_samples):
    distance_fn = lambda x : sklearn.metrics.pairwise.cosine_distances(x[0],x)[0] * 100
    features = x.nonzero()[1]
    vals = np.array(x[x.nonzero()])[0]
    doc_size = len(sp.sparse.find(x)[2])                                    
    sample = np.random.randint(1, doc_size, num_samples - 1)                             
    data = np.zeros((num_samples, len(features)))    
    inverse_data = np.zeros((num_samples, len(features)))                                         
    data[0] = np.ones(doc_size)
    inverse_data[0] = vals
    features_range = range(len(features)) 
    for i, s in enumerate(sample, start=1):                                               
        active = np.random.choice(features_range, s, replace=False)                       
        data[i, active] = 1
        for j in active:
            inverse_data[i, j] = 1
    sparse_inverse = sp.sparse.lil_matrix((inverse_data.shape[0], x.shape[1]))
    sparse_inverse[:, features] = inverse_data
    sparse_inverse = sp.sparse.csr_matrix(sparse_inverse)
    mapping = features
    labels = classifier_fn(sparse_inverse)
    distances = distance_fn(sparse_inverse)
    return data, labels, distances, mapping

# This is LIME
class GeneralizedLocalExplainer:
  def __init__(self,
               kernel_fn,
               data_labels_distances_mapping_fn,
               num_samples=5000,
               lasso=True,
               mean=None,
               return_mean=False,
               return_mapped=False,
               lambda_=None,
               verbose=True,
               positive=False):
    # Transform_classifier, transform_explainer,
    # transform_explainer_to_classifier all take raw data in, whatever that is.
    # perturb(x, num_samples) returns data (perturbed data in f'(x) form),
    # inverse_data (perturbed data in x form) and mapping, where mapping is such
    # that mapping[i] = j, where j is an index for x form.
    # distance_fn takes raw data in. what we're calling raw data is just x
    self.lambda_ = lambda_
    self.kernel_fn = kernel_fn
    self.data_labels_distances_mapping_fn = data_labels_distances_mapping_fn
    self.num_samples = num_samples
    self.lasso = lasso
    self.mean = mean
    self.return_mapped=return_mapped
    self.return_mean = return_mean
    self.verbose = verbose
    self.positive=positive;
  def reset(self):
    pass
  def data_labels_distances_mapping(self, raw_data, classifier_fn):
    data, labels, distances, mapping = self.data_labels_distances_mapping_fn(raw_data, classifier_fn, self.num_samples)
    return data, labels, distances, mapping
  def generate_lars_path(self, weighted_data, weighted_labels):
    X = weighted_data
    alphas, active, coefs = linear_model.lars_path(X, weighted_labels, method='lasso', verbose=False, positive=self.positive)
    return alphas, coefs
  def explain_instance_with_data(self, data, labels, distances, label, num_features):
    weights = self.kernel_fn(distances)
    weighted_data = data * weights[:, np.newaxis]
    if self.mean is None:
      mean = np.mean(labels[:, label])
    else:
      mean = self.mean
    shifted_labels = labels[:, label] - mean
    if self.verbose:
      print 'mean', mean
    weighted_labels = shifted_labels * weights
    used_features = range(weighted_data.shape[1])
    nonzero = used_features
    alpha = 1
    if self.lambda_:
      classif = linear_model.Lasso(alpha=self.lambda_, fit_intercept=False, positive=self.positive)
      classif.fit(weighted_data, weighted_labels)
      used_features = classif.coef_.nonzero()[0]
      if used_features.shape[0] == 0:
        if self.return_mean:
          return [], mean
        else:
          return []
    elif self.lasso:
      alphas, coefs = self.generate_lars_path(weighted_data, weighted_labels)
      for i in range(len(coefs.T) - 1, 0, -1):
        nonzero = coefs.T[i].nonzero()[0]
        if len(nonzero) <= num_features:
            chosen_coefs = coefs.T[i]
            alpha = alphas[i]
            break
      used_features = nonzero
    debiased_model = linear_model.Ridge(alpha=0, fit_intercept=False)
    debiased_model.fit(weighted_data[:, used_features], weighted_labels)
    if self.verbose:
      print 'Prediction_local', debiased_model.predict(data[0, used_features].reshape(1, -1)) + mean, 'Right:', labels[0, label]
    if self.return_mean:
      return sorted(zip(used_features,
                  debiased_model.coef_),
                  key=lambda x:np.abs(x[1]), reverse=True), mean
    else:
      return sorted(zip(used_features,
                  debiased_model.coef_),
                  key=lambda x:np.abs(x[1]), reverse=True)

  def explain_instance(self,
                       raw_data,
                       label,
                       classifier_fn,
                       num_features, dataset=None):
    
    if not hasattr(classifier_fn, '__call__'):
      classifier_fn = classifier_fn.predict_proba
    data, labels, distances, mapping = self.data_labels_distances_mapping(raw_data, classifier_fn)
    if self.return_mapped:
      if self.return_mean:
        exp, mean =   self.explain_instance_with_data(data, labels, distances, label, num_features)
      else:
        exp =   self.explain_instance_with_data(data, labels, distances, label, num_features)
      exp = [(mapping[x[0]], x[1]) for x in exp]
      if self.return_mean:
        return exp, mean
      else:
        return exp
    return self.explain_instance_with_data(data, labels, distances, label, num_features), mapping


