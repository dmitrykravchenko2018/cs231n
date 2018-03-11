import numpy as np

# a = np.arange(2 * 10).reshape(2, 10)
#
# b = np.arange(20 * 10).reshape(20, 10)


# a_square_sum = np.sum(np.square(a), axis=1)  # (1 x num_test) or (num_test,)
# b_square_sum = np.sum(np.square(b), axis=1)  # (1 x num_train) or (num _train,)
#
# inner_product = np.dot(a, b.T)  # (num_test x num_train)
# print("Inner product a*b", inner_product.shape)
# #  print(b_square_sum, b_square_sum.reshape(-1, 1))
# print(b_square_sum.shape)
# l2 = np.sqrt(-2 * inner_product + a_square_sum.reshape(-1, 1) + b_square_sum)
# print(l2)

# X_train = np.arange(300).reshape(50, 6)
# y_train = np.arange(50)
# print("X train :\n", X_train)
# print("y train labels :\n", y_train)
# print("------------------\n")
# X_train_folds = np.array_split(X_train, 5, axis=0)
# y_train_folds = np.array_split(y_train, 5, axis=0)
# print(X_train_folds, y_train_folds)
# print("------------------\n")
# fold = X_train_folds[0]
# print(fold.shape, type(fold))

x = np.arange(2 * 20).reshape(2, 20)

dw = np.arange(20 * 10).reshape(20, 10)

print(x[1, :].shape, x[1, :])
print(dw[:, 1].shape, dw[:, 1])

# print(np.add(dw[:, 1], x[1, :].T))
dw[:, 1] += x[1, :].T
print(dw)
