import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

with open('./t10k-images.idx3-ubyte', 'rb') as fp :
    magic, numspl, n1, n2 = np.fromfile(fp, dtype='>i4', count=4)
    assert magic == 2051

    data = np.fromfile(fp, dtype='>u1').astype('f4').reshape(-1, n1*n2)

kw = dict(figsize=(8, 8))

# 1) show the data
fig, ax = plt.subplots(**kw)
ax.matshow(data[:1024])
plt.show()

# 2) perform t-SNE
proj = TSNE(
    n_components=2,
    max_iter=600,
    verbose=5
).fit_transform(data)

# 3) show the t-SNE embedding
fig, ax = plt.subplots(**kw)
ax.scatter(*proj.T)
plt.show()

# 4) different view on the data
fig, ax = plt.subplots(nrows=4, ncols=4, **kw)
ax = ax.flatten()
for ii in range(16) :
    ax[ii].matshow(data[ii].reshape(n1, n2))
plt.show()

with open('./t10k-labels.idx1-ubyte', 'rb') as fp :
    magic, _numspl = np.fromfile(fp, dtype='>i4', count=2)
    assert magic == 2049
    assert numspl == _numspl

    labels = np.fromfile(fp, dtype='>u1').astype('f4')

# 5) t-SNE embedding with labels
fig, ax = plt.subplots(**kw)
ax.scatter(*proj.T, c=labels)
plt.show()
