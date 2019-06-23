import re
import torch
import numpy as np
def load_embedding(embedding_path,words):
    if words is not None:
        words_set = set()
        idx_dict = {}
        for idx,w in words.items():
            words_set.add(w)
            idx_dict[w] = int(idx)

    vectors = np.zeros((len(words)+1,300))

    num_oovs = 0

    with open(embedding_path, encoding="utf-8") as fp:

        row1 = fp.readline()
        # if the first row is not header
        if not re.match('^[0-9]+ [0-9]+$', row1):
            # seek to 0
            fp.seek(0)
        # otherwise ignore the header

        for i, line in enumerate(fp):
            cols = line.rstrip().split(' ')
            word = cols[0]

            # skip word not in words if words are provided
            if word not in words_set:
                # vectors.append(oov)
                num_oovs += 1
            elif word in words_set:
                _id = idx_dict[word]
                vectors[_id] = np.array([float(v) for v in cols[1:]])


    vectors = torch.tensor(vectors, dtype=torch.float32)
    print(num_oovs)
    print("fasttext embedding size",vectors.size())
    return vectors
