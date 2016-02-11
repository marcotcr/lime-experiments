## Experiment in section 5.2:
dataset can be 'multi_polarity_books', 'multi_polarity_kitchen', 'multi_polarity_dvd', 'multi_polarity_kitchen'
algorithm can be 'l1logreg', 'tree'
explainer can be 'lime', 'greedy' or 'random' (parzen requires additional code)

    python evaluate_explanations.py --dataset DATASET --algorithm ALGORITHM --explainer EXPLAINER 

## Experiment in section 5.3:
  algorithm can be 'logreg', 'random_forest', 'svm', 'tree' or 'embforest', although you would need to set up word2vec for embforest

    python2.7 data_trusting.py -d dataset -a algorithm -k 10 -u .1 -r NUM_ROUNDS
