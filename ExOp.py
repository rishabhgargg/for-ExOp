#Solution for ExOp for AI in Kochi


import numpy as np
import pandas as pd
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

file_frame = pd.read_csv('train.csv')

def compute_score(row):
    "Computes Intersection over Union score between question1 and question2"
    q1_list = []
    q2_list = []
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1_list.append(word)
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2_list.append(word)
    num = len(set(q1_list) & set(q2_list))
    deno = len(set(q1_list) | set(q2_list))
    if len(q1_list) == 0 or len(q2_list) == 0:
        return 0
    score = float(num)/float(deno)
    return score


score_list = []
for index, row in file_frame.iterrows():
    score = compute_score(row)
    score_list.append(score)

file_frame['intersection_over_union_score'] = score_list

A = np.array([score_list, np.ones(len(score_list))])

w = np.linalg.lstsq(A.T,file_frame['is_duplicate'])[0]


test_frame = pd.read_csv('test.csv')

score_list = []
for index, row in test_frame.iterrows():
    score = compute_score(row)
    score_list.append(score)

test_frame['intersection_over_union_score'] = score_list

test_frame['is_duplicate'] = test_frame['intersection_over_union_score']*w[0]+w[1]


sub = pd.DataFrame()
sub['test_id'] = test_frame['test_id']
sub['is_duplicate'] = test_frame['is_duplicate']
sub.to_csv('rishabh_exop_sol.csv', index=False)
