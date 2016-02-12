import sys
import copy
import os
import numpy as np
import scipy as sp
import json
import random
import sklearn
from sklearn import ensemble
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
import pickle
import explainers
import parzen_windows
import embedding_forest
from load_datasets import *
import argparse
import collections
    
def get_classifier(name, vectorizer):
  if name == 'logreg':
    return linear_model.LogisticRegression(fit_intercept=True)
  if name == 'random_forest':
    return ensemble.RandomForestClassifier(n_estimators=1000, random_state=1, max_depth=5, n_jobs=10)
  if name == 'svm':
    return svm.SVC(probability=True, kernel='rbf', C=10,gamma=0.001)
  if name == 'tree':
    return tree.DecisionTreeClassifier(random_state=1)
  if name == 'neighbors':
    return neighbors.KNeighborsClassifier()
  if name == 'embforest':
    return embedding_forest.EmbeddingForest(vectorizer)

def main():
  parser = argparse.ArgumentParser(description='Evaluate some explanations')
  parser.add_argument('--dataset', '-d', type=str, required=True,help='dataset name')
  parser.add_argument('--algorithm', '-a', type=str, required=True, help='algorithm_name')
  parser.add_argument('--num_features', '-k', type=int, required=True, help='num features')
  parser.add_argument('--percent_untrustworthy',  '-u', type=float, required=True, help='percentage of untrustworthy features. like 0.1')
  parser.add_argument('--num_rounds', '-r', type=int, required=True, help='num rounds')
  args = parser.parse_args()
  dataset = args.dataset
  train_data, train_labels, test_data, test_labels, class_names = LoadDataset(dataset)
  vectorizer = CountVectorizer(lowercase=False, binary=True) 
  train_vectors = vectorizer.fit_transform(train_data)
  test_vectors = vectorizer.transform(test_data)
  terms = np.array(list(vectorizer.vocabulary_.keys()))
  indices = np.array(list(vectorizer.vocabulary_.values()))
  inverse_vocabulary = terms[np.argsort(indices)]

  np.random.seed(1)
  classifier = get_classifier(args.algorithm, vectorizer)
  classifier.fit(train_vectors, train_labels)


  np.random.seed(1)
  untrustworthy_rounds = []
  all_features = range(train_vectors.shape[1])
  num_untrustworthy = int(train_vectors.shape[1] * args.percent_untrustworthy)
  for _ in range(args.num_rounds):
    untrustworthy_rounds.append(np.random.choice(all_features, num_untrustworthy, replace=False))
  
  rho = 25
  kernel = lambda d: np.sqrt(np.exp(-(d**2) / rho ** 2))
  LIME = explainers.GeneralizedLocalExplainer(kernel, explainers.data_labels_distances_mapping_text, num_samples=15000, return_mean=True, verbose=False, return_mapped=True)

  parzen = parzen_windows.ParzenWindowClassifier()
  cv_preds = sklearn.cross_validation.cross_val_predict(classifier, train_vectors, train_labels, cv=5)
  parzen.fit(train_vectors, cv_preds)
  sigmas = {'multi_polarity_electronics': {'neighbors': 0.75, 'svm': 10.0, 'tree': 0.5,
  'logreg': 0.5, 'random_forest': 0.5, 'embforest': 0.75},
  'multi_polarity_kitchen': {'neighbors': 1.0, 'svm': 6.0, 'tree': 0.75,
  'logreg': 0.25, 'random_forest': 6.0, 'embforest': 1.0},
  'multi_polarity_dvd': {'neighbors': 0.5, 'svm': 0.75, 'tree': 8.0, 'logreg':
  0.75, 'random_forest': 0.5, 'embforest': 5.0}, 'multi_polarity_books':
  {'neighbors': 0.5, 'svm': 7.0, 'tree': 2.0, 'logreg': 1.0, 'random_forest':
  1.0, 'embforest': 3.0}}
  parzen.sigma = sigmas[dataset][args.algorithm]

  random = explainers.RandomExplainer()
  exps = {}
  explainer_names = ['LIME', 'random', 'greedy', 'parzen']
  for expl in explainer_names:
    exps[expl] = []

  predictions = classifier.predict(test_vectors)
  predict_probas = classifier.predict_proba(test_vectors)[:,1]
  for i in range(test_vectors.shape[0]):
    print i
    sys.stdout.flush()
    exp, mean = LIME.explain_instance(test_vectors[i], 1, classifier.predict_proba, args.num_features)
    exps['LIME'].append((exp, mean))
    exp = parzen.explain_instance(test_vectors[i], 1, classifier.predict_proba, args.num_features, None) 
    mean = parzen.predict_proba(test_vectors[i])[1]
    exps['parzen'].append((exp, mean))

    exp = random.explain_instance(test_vectors[i], 1, None, args.num_features, None)
    exps['random'].append(exp)

    exp = explainers.explain_greedy_martens(test_vectors[i], predictions[i], classifier.predict_proba, args.num_features)
    exps['greedy'].append(exp)

  precision = {}
  recall = {}
  f1 = {}
  for name in explainer_names:
    precision[name] = []
    recall[name] = []
    f1[name] = []
  flipped_preds_size = []
  for untrustworthy in untrustworthy_rounds:
    t = test_vectors.copy()
    t[:, untrustworthy] = 0
    mistrust_idx = np.argwhere(classifier.predict(t) != classifier.predict(test_vectors)).flatten()
    print 'Number of suspect predictions', len(mistrust_idx)
    shouldnt_trust = set(mistrust_idx)
    flipped_preds_size.append(len(shouldnt_trust))
    mistrust = collections.defaultdict(lambda:set())
    trust = collections.defaultdict(lambda: set())
    trust_fn = lambda prev, curr: (prev > 0.5 and curr > 0.5) or (prev <= 0.5 and curr <= 0.5)
    trust_fn_all = lambda exp, unt: len([x[0] for x in exp if x[0] in unt]) == 0
    for i in range(test_vectors.shape[0]):
      exp, mean = exps['LIME'][i]
      prev_tot = predict_probas[i]
      prev_tot2 = sum([x[1] for x in exp]) + mean
      tot = prev_tot2 - sum([x[1] for x in exp if x[0] in untrustworthy])
      trust['LIME'].add(i) if trust_fn(tot, prev_tot) else mistrust['LIME'].add(i)

      exp, mean = exps['parzen'][i]
      prev_tot = mean
      tot = mean - sum([x[1] for x in exp if x[0] in untrustworthy])
      trust['parzen'].add(i) if trust_fn(tot, prev_tot) else mistrust['parzen'].add(i)
      exp = exps['random'][i]
      trust['random'].add(i) if trust_fn_all(exp, untrustworthy) else mistrust['random'].add(i)

      exp = exps['greedy'][i]
      trust['greedy'].add(i) if trust_fn_all(exp, untrustworthy) else mistrust['greedy'].add(i)

    for expl in explainer_names:
      # switching the definition
      false_positives = set(trust[expl]).intersection(shouldnt_trust)
      true_positives = set(trust[expl]).difference(shouldnt_trust)
      false_negatives = set(mistrust[expl]).difference(shouldnt_trust)
      true_negatives = set(mistrust[expl]).intersection(shouldnt_trust)

      try:
        prec= len(true_positives) / float(len(true_positives) + len(false_positives))
      except:
        prec= 0
      try:
        rec= float(len(true_positives)) / (len(true_positives) + len(false_negatives))
      except:
        rec= 0
      precision[expl].append(prec)
      recall[expl].append(rec)
      f1z = 2 * (prec * rec) / (prec + rec) if (prec and rec) else 0
      f1[expl].append(f1z)

  print 'Average number of flipped predictions:', np.mean(flipped_preds_size), '+-', np.std(flipped_preds_size)
  print 'Precision:'
  for expl in explainer_names:
    print expl, np.mean(precision[expl]), '+-', np.std(precision[expl]), 'pvalue', sp.stats.ttest_ind(precision[expl], precision['LIME'])[1].round(4)
  print
  print 'Recall:'
  for expl in explainer_names:
    print expl, np.mean(recall[expl]), '+-', np.std(recall[expl]), 'pvalue', sp.stats.ttest_ind(recall[expl], recall['LIME'])[1].round(4)
  print 
  print 'F1:'
  for expl in explainer_names:
    print expl, np.mean(f1[expl]), '+-', np.std(f1[expl]), 'pvalue', sp.stats.ttest_ind(f1[expl], f1['LIME'])[1].round(4)
if __name__ == "__main__":
    main()
