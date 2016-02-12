import sys
import copy
sys.path.append('..')
import time
import numpy as np
import scipy as sp
import scipy.stats
import sklearn
import xgboost
import xgboost.sklearn
import explainers
from load_datasets import *
import glob
import argparse
import collections
from sklearn import ensemble, cross_validation
import pickle
import parzen_windows

def submodular_fn(explanations, feature_value):
  """TODO: Detail this"""
  z_words = set()
  for exp in explanations:
    z_words = z_words.union([x[0] for x in exp])
  normalizer = sum([feature_value[w] for w in z_words])
  def fnz(x):
    all_words = set()
    for doc in x:
      all_words = all_words.union([x[0] for x in explanations[doc]])
    return sum([feature_value[w] for w in all_words]) / normalizer
  fnz.num_items = len(explanations)
  return fnz
def greedy(submodular_fn, k, chosen=[]):
    chosen = copy.deepcopy(chosen)
    all_items = range(submodular_fn.num_items)
    current_value = 0
    while len(chosen) != k:
        best_gain = 0
        best_item = all_items[0]
        for i in all_items:
            gain= submodular_fn(chosen + [i]) - current_value
            if gain > best_gain:
                best_gain = gain
                best_item = i
        chosen.append(best_item)
        all_items.remove(best_item)
        current_value += best_gain 
    return chosen
# A pick function takes in the whole map, returns two lists of tuples with instance
# ids and weights, one for each classifier. This won't work later if I want it to be interactive.
def submodular_pick(pickled_map, explainer, B, use_explanation_weights=False,
alternate=False):
  def get_function(exps):
    feature_value = collections.defaultdict(float)
    for exp in exps:
      for f, v in exp:
        if not use_explanation_weights:
          v = 1
        feature_value[f] += np.abs(v)
    for f in feature_value:
        feature_value[f] = np.sqrt(feature_value[f])
    submodular = submodular_fn(exps, feature_value)
    return submodular
    out = greedy(submodular, B)
    return out
  if explainer in ['parzen', 'lime']:
    exps1 = [x[0] for x in pickled_map['exps1'][explainer]]
    exps2 = [x[0] for x in pickled_map['exps2'][explainer]]
  else:
    exps1 = pickled_map['exps1'][explainer]
    exps2 = pickled_map['exps2'][explainer]
  fn1 = get_function(exps1)
  fn2 = get_function(exps2)
  if not alternate:
    return greedy(fn1, B), greedy(fn2, B)
  else:
    ret = []
    for i in range(B):
      fn = fn1 if i % 2 == 0 else fn2
      ret = greedy(fn, i + 1, ret)
    return ret 
      
  #return get_list(exps1), get_list(exps2)

def all_pick(pickled_map, explainer, B):
  list_ = range(len(pickled_map['exps1'][explainer]))
  return list_, list_

def random_pick(pickled_map, explainer, B):
  list_ = np.random.choice(range(len(pickled_map['exps1'][explainer])), B, replace=False)
  return list_, list_

def find_untrustworthy(explainer, exps, instances, untrustworthy):
  found = set()
  for i in instances:
    if explainer in ['lime', 'parzen']:
      exp, mean = exps[i]
    else:
      exp = exps[i]
    found = found.union([x[0] for x in exp if x[0] in untrustworthy])
  return found

def tally_mistrust(explainer, exps, predict_probas, untrustworthy):
  trust_fn = lambda prev, curr: (prev > 0.5 and curr > 0.5) or (prev <= 0.5 and curr <= 0.5)
  trust_fn_all = lambda exp, unt: len([x[0] for x in exp if x[0] in unt]) == 0
  mistrust = 0
  for i in range(len(exps)):
    if explainer in ['lime', 'parzen']:
      exp, mean = exps[i]
      if explainer == 'lime':
        prev_tot = sum([x[1] for x in exp]) + mean
      elif explainer == 'parzen':
        prev_tot = mean
      tot = prev_tot - sum([x[1] for x in exp if x[0] in untrustworthy])
      if not trust_fn(tot, prev_tot):
        mistrust += 1
    else:
      exp = exps[i]
      if not trust_fn_all(exp, untrustworthy):
        mistrust += 1
  return mistrust 


def main():
  parser = argparse.ArgumentParser(description='Evaluate some explanations')
  parser.add_argument('--dataset', '-d', type=str, required=True,help='dataset name')
  parser.add_argument('--output_folder', '-o', type=str, required=True, help='output folder')
  parser.add_argument('--num_features', '-k', type=int, required=True, help='num features')
  parser.add_argument('--pick', '-p', type=str, required=False, default='all', help='all, submodular, submodular2 or random')
  parser.add_argument('--num_instances', '-n', type=int, required=False, default=1, help='number of instances to look at')
  parser.add_argument('--num_rounds', '-r', type=int, required=False, default=10, help='num rounds')
  #parser.add_argument('--start_id',  '-i', type=int, required=True, help='output start id')
  args = parser.parse_args()
  dataset = args.dataset
  got_right = lambda test1, test2, mistrust1, mistrust2: mistrust1 < mistrust2 if test1 > test2 else mistrust1 > mistrust2
  names = ['lime', 'parzen', 'random', 'greedy']
  num_exps = 0
  B = args.num_instances
  rounds = 1
  if args.pick == 'all':
    pick_function = all_pick
  elif args.pick == 'submodular':
    pick_function = lambda a,b,c : submodular_pick(a,b,c, use_explanation_weights=True)
  elif args.pick == 'random':
    pick_function = random_pick
    rounds =args.num_rounds
  accuracy = collections.defaultdict(lambda: [])
  right = collections.defaultdict(lambda: [])
  for r in range(rounds):
    right = collections.defaultdict(lambda: [])
    for filez in glob.glob(os.path.join(args.output_folder, 'comparing_%s*' % args.dataset))[:800]:
      num_exps += 1
      pickled_map = pickle.load(open(filez))
      predict_probas = pickled_map['predict_probas1']
      predict_probas2 = pickled_map['predict_probas2']
      test1 = pickled_map['test_acc1']
      test2 = pickled_map['test_acc2']
      untrustworthy = pickled_map['untrustworthy']
      for explainer in names:
        if explainer.startswith('lime'):
          pick1, pick2 = pick_function(pickled_map, 'lime', B)
          exps1 = pickled_map['exps1']['lime']
          exps2 = pickled_map['exps2']['lime']
        elif explainer.startswith('parzen'):
          pick1, pick2 = pick_function(pickled_map, 'parzen', B)
          exps1 = pickled_map['exps1']['parzen']
          exps2 = pickled_map['exps2']['parzen']
        else:
          pick1, pick2 = pick_function(pickled_map, explainer, B)
          exps1 = pickled_map['exps1'][explainer]
          exps2 = pickled_map['exps2'][explainer]
        if args.pick != 'all':
          unt1 = find_untrustworthy(explainer, exps1, pick1, untrustworthy)
          unt2 = find_untrustworthy(explainer, exps2, pick2, untrustworthy)
        else:
          unt1 = unt2 = untrustworthy
        mistrust1 = tally_mistrust(explainer, exps1, predict_probas, unt1)
        mistrust2 = tally_mistrust(explainer, exps2, predict_probas2, unt2)
        while mistrust1 == mistrust2:
          mistrust1 = np.random.randint(0,10)                                             
          mistrust2 = np.random.randint(0,10)
        #print explainer, mistrust1, mistrust2
        right[explainer].append(int(got_right(test1, test2, mistrust1, mistrust2)))
      right['random_choice'].append(int(got_right(test1, test2, np.random.random(), np.random.random())))
      #print [(x[0], sum(x[1])) for x in right.iteritems()]
      #print filez
    for name in right:
      accuracy[name].append(np.mean(right[name]))
  print 'Mean accuracy:'
  for name in right:
    print name, np.mean(accuracy[name])


if __name__ == "__main__":
    main()
