import sys
import copy
sys.path.append('..')
import time
import numpy as np
import scipy as sp
import sklearn
import xgboost
import xgboost.sklearn
import explainers
from load_datasets import *
from sklearn.metrics import accuracy_score
from sklearn import ensemble, cross_validation
import pickle
import parzen_windows
import argparse
def get_random_indices(labels, class_, probability):
  nonzero = (labels == class_).nonzero()[0]
  if nonzero.shape[0] == 0 or probability == 0:
    return []
  return np.random.choice(nonzero, int(probability * len(nonzero)) , replace=False)
def add_corrupt_feature(feature_name, clean_train, clean_test, dirty_train,
                        train_labels, test_labels, class_probs_dirty, class_probs_clean, fake_prefix='FAKE'):
    """clean_train, clean_test, dirty_train will be corrupted"""
    for class_ in set(train_labels):
      indices = get_random_indices(train_labels, class_, class_probs_clean[class_])
      for i in indices:
        clean_train[i] += ' %s%s%s' % (fake_prefix, feature_name, fake_prefix)
      indices = get_random_indices(train_labels, class_, class_probs_dirty[class_])
      for i in indices:
        dirty_train[i] += ' %s%s%s' % (fake_prefix, feature_name, fake_prefix)
      indices = get_random_indices(test_labels, class_, class_probs_clean[class_])
      for i in indices:
        clean_test[i] += ' %s%s%s' % (fake_prefix, feature_name, fake_prefix)
def corrupt_dataset(independent_features, train_data, train_labels, test_data, test_labels):
    # independent_features: list [([.3, .8],[.5,.5], 3), ([.1, .1],[0, 0], 1)
    # ...]. Each element in list is a tuple (l,l2, n) where l a list
    # representing the probability of seeing the feature in each class in the
    # dirty train data, l2 is a list representing the probability of seeing the
    # feature in each class the clean test data and n is the number of features
    # with this distribution to add.
    # returns (clean_train, dirty_train, clean_test)
    dirty_train = copy.deepcopy(train_data)
    clean_train = copy.deepcopy(train_data)
    clean_test = copy.deepcopy(test_data)
    idx = 0
    for probs, probs2, n in independent_features:
        for i in range(n):
            add_corrupt_feature('%d' % idx, clean_train, clean_test, dirty_train, train_labels, test_labels, probs, probs2)
            idx += 1
    return clean_train, dirty_train, clean_test
def main():
  parser = argparse.ArgumentParser(description='Evaluate some explanations')
  parser.add_argument('--dataset', '-d', type=str, required=True,help='dataset name')
  parser.add_argument('--output_folder', '-o', type=str, required=True, help='output folder')
  parser.add_argument('--num_features', '-k', type=int, required=True, help='num features')
  parser.add_argument('--num_rounds', '-r', type=int, required=True, help='num rounds')
  parser.add_argument('--start_id',  '-i', type=int, default=0,required=False, help='output start id')
  args = parser.parse_args()
  dataset = args.dataset
  train_data, train_labels, test_data, test_labels, class_names = LoadDataset(dataset)
  rho = 25
  kernel = lambda d: np.sqrt(np.exp(-(d**2) / rho ** 2))
  local = explainers.GeneralizedLocalExplainer(kernel, explainers.data_labels_distances_mapping_text, num_samples=15000, return_mean=True, verbose=False, return_mapped=True)
  # Found through cross validation
  sigmas = {'multi_polarity_electronics': {'neighbors': 0.75, 'svm': 10.0, 'tree': 0.5,
  'logreg': 0.5, 'random_forest': 0.5, 'embforest': 0.75},
  'multi_polarity_kitchen': {'neighbors': 1.0, 'svm': 6.0, 'tree': 0.75,
  'logreg': 0.25, 'random_forest': 6.0, 'embforest': 1.0},
  'multi_polarity_dvd': {'neighbors': 0.5, 'svm': 0.75, 'tree': 8.0, 'logreg':
  0.75, 'random_forest': 0.5, 'embforest': 5.0}, 'multi_polarity_books':
  {'neighbors': 0.5, 'svm': 7.0, 'tree': 2.0, 'logreg': 1.0, 'random_forest':
  1.0, 'embforest': 3.0}}
  parzen1 = parzen_windows.ParzenWindowClassifier()
  parzen1.sigma = sigmas[dataset]['random_forest']
  parzen2 = parzen_windows.ParzenWindowClassifier()
  parzen2.sigma = sigmas[dataset]['random_forest']
  random = explainers.RandomExplainer()

  for Z in range(args.num_rounds):
    exps1 = {}
    exps2 = {}
    explainer_names = ['lime', 'parzen', 'random', 'greedy', 'mutual']
    for expl in explainer_names:
      exps1[expl] = []
      exps2[expl] = []
    print 'Round', Z
    sys.stdout.flush()
    fake_features_z = [([.1, .2], [.1,.1], 10)]#, ([.2, .1], [.1,.1], 10)]
    clean_train, dirty_train, clean_test = corrupt_dataset(fake_features_z, train_data, train_labels, test_data, test_labels)
    vectorizer = CountVectorizer(lowercase=False, binary=True) 
    dirty_train_vectors = vectorizer.fit_transform(dirty_train)
    clean_train_vectors = vectorizer.transform(clean_train)
    test_vectors = vectorizer.transform(clean_test)
    terms = np.array(list(vectorizer.vocabulary_.keys()))
    indices = np.array(list(vectorizer.vocabulary_.values()))
    inverse_vocabulary = terms[np.argsort(indices)]
    tokenizer = vectorizer.build_tokenizer()  
    c1 = ensemble.RandomForestClassifier(n_estimators=30, max_depth=5)
    c2 = ensemble.RandomForestClassifier(n_estimators=30, max_depth=5)
    untrustworthy = [i for i, x in enumerate(inverse_vocabulary) if x.startswith('FAKE')]
    train_idx, test_idx = tuple(cross_validation.ShuffleSplit(dirty_train_vectors.shape[0], 1, 0.2))[0]
    train_acc1 = train_acc2 = test_acc1 = test_acc2 = 0
    print 'Trying to find trees:'
    sys.stdout.flush()
    iteration = 0
    found_tree = True
    while np.abs(train_acc1 - train_acc2) > 0.001 or np.abs(test_acc1 - test_acc2) < 0.05: 
      iteration += 1
      c1.fit(dirty_train_vectors[train_idx], train_labels[train_idx])
      c2.fit(dirty_train_vectors[train_idx], train_labels[train_idx])
      train_acc1 = accuracy_score(train_labels[test_idx], c1.predict(dirty_train_vectors[test_idx]))
      train_acc2 = accuracy_score(train_labels[test_idx], c2.predict(dirty_train_vectors[test_idx]))
      test_acc1 = accuracy_score(test_labels, c1.predict(test_vectors))
      test_acc2 = accuracy_score(test_labels, c2.predict(test_vectors))
      if iteration == 3000:
        found_tree = False
        break
    if not found_tree:
      print 'skipping iteration', Z
      continue
    print 'done'
    print 'Train acc1:', train_acc1, 'Train acc2:', train_acc2
    print 'Test acc1:', test_acc1, 'Test acc2:', test_acc2
    sys.stdout.flush()
    predictions = c1.predict(dirty_train_vectors)
    predictions2 = c2.predict(dirty_train_vectors)
    predict_probas = c1.predict_proba(dirty_train_vectors)[:,1]
    predict_probas2 = c2.predict_proba(dirty_train_vectors)[:,1]
    cv_preds1 = cross_validation.cross_val_predict(c1, dirty_train_vectors[train_idx], train_labels[train_idx], cv=5)
    cv_preds2 = cross_validation.cross_val_predict(c2, dirty_train_vectors[train_idx], train_labels[train_idx], cv=5)
    parzen1.fit(dirty_train_vectors[train_idx], cv_preds1)
    parzen2.fit(dirty_train_vectors[train_idx], cv_preds2)
    pp = []
    pp2 = []
    true_labels = []
    iteration = 0
    for i in test_idx:
      if iteration % 50 == 0:
        print iteration
        sys.stdout.flush()
      iteration += 1
      pp.append(predict_probas[i])
      pp2.append(predict_probas2[i])
      true_labels.append(train_labels[i])
      exp, mean = local.explain_instance(dirty_train_vectors[i], 1, c1.predict_proba, args.num_features)
      exps1['lime'].append((exp, mean))

      exp = parzen1.explain_instance(dirty_train_vectors[i], 1, c1.predict_proba, args.num_features, None) 
      mean = parzen1.predict_proba(dirty_train_vectors[i])[1]
      exps1['parzen'].append((exp, mean))

      exp = random.explain_instance(dirty_train_vectors[i], 1, None, args.num_features, None)
      exps1['random'].append(exp)

      exp = explainers.explain_greedy_martens(dirty_train_vectors[i], predictions[i], c1.predict_proba, args.num_features)
      exps1['greedy'].append(exp)


      # Classifier 2
      exp, mean = local.explain_instance(dirty_train_vectors[i], 1, c2.predict_proba, args.num_features)
      exps2['lime'].append((exp, mean))

      exp = parzen2.explain_instance(dirty_train_vectors[i], 1, c2.predict_proba, args.num_features, None) 
      mean = parzen2.predict_proba(dirty_train_vectors[i])[1]
      exps2['parzen'].append((exp, mean))

      exp = random.explain_instance(dirty_train_vectors[i], 1, None, args.num_features, None)
      exps2['random'].append(exp)

      exp = explainers.explain_greedy_martens(dirty_train_vectors[i], predictions2[i], c2.predict_proba, args.num_features)
      exps2['greedy'].append(exp)

    out = {'true_labels' : true_labels, 'untrustworthy' : untrustworthy, 'train_acc1' :  train_acc1, 'train_acc2' : train_acc2, 'test_acc1' : test_acc1, 'test_acc2' : test_acc2, 'exps1' : exps1, 'exps2': exps2, 'predict_probas1': pp, 'predict_probas2': pp2}
    pickle.dump(out, open(os.path.join(args.output_folder, 'comparing_%s_%s_%d.pickle' % (dataset, args.num_features, Z + args.start_id)), 'w'))



if __name__ == "__main__":
    main()
