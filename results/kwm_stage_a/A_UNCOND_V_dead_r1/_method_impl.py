import pandas as pd, numpy as np
df = pd.read_csv('train_v1.csv')
sub = df.dropna(subset=['y'])
a = sub[sub.group=='A'].y; b = sub[sub.group=='B'].y
d = a.mean()-b.mean(); nA, nB = len(a), len(b)
sp = np.sqrt(((nA-1)*a.var()+(nB-1)*b.var())/(nA+nB-2))
se = sp*np.sqrt(1/nA+1/nB)
print('DELTA %.4f' % d)
print('CI95 [%.4f, %.4f]' % (d-1.96*se, d+1.96*se))
print('N %d' % len(sub))
