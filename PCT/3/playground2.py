import numpy as np
a = '12 13 14'
b = {'a':3, 'b':4}

emb0 = [1, 2, 3]
emd1 = [4,5,6]
emb3 = [7,8,9]

c = np.hstack([emb3, emb0, emd1])

print(c)