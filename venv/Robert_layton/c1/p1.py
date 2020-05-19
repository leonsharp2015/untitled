import numpy as np
from collections import defaultdict
X=np.loadtxt('affinity_dataset.txt')
n_samples, n_features = X.shape

valid_rules=defaultdict(int)
invalid_rules=defaultdict(int)
num_occurences=defaultdict(int)
num_apple_purchases = 0

for sample in X:
    for premise in range(n_features):
        if sample[premise] == 0: continue
        num_occurences[premise] += 1
        for conclusion in range(n_features):
            if premise == conclusion:
                continue
            if sample[conclusion] == 1:
                valid_rules[(premise, conclusion)] += 1
            else:
                invalid_rules[(premise, conclusion)] += 1
support = valid_rules
confidence = defaultdict(float)
for premise, conclusion in valid_rules.keys():
    confidence[(premise, conclusion)] = valid_rules[(premise, conclusion)] / num_occurences[premise]











