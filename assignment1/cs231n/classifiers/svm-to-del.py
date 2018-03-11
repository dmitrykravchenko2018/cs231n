import numpy as np

# y = np.array([3, 2, 3, 3, 3, 3, 3, 3, 3, 3])
# scores = np.arange(10*5).reshape(10, 5)
# print("--------\n", scores, "\n-----------\n")
# print(y, scores - scores[:, 3] + 1)
#
# scores = np.arange(5)
# for i in range(scores.shape[0]):
#     margins = scores - scores[3] + 1
#
# print("L i vectorized\n", margins, margins.shape)

y = np.arange(32)
batch_size = 8
x = np.arange(32*10).reshape(32, 10)
y_batch = np.random.choice(y, 8)
print(x, y_batch)
x_batch = x[y_batch, :]
print(x_batch)