import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    # Use trick with constant to improve numerical stability of computation
    # common choice logC = - max_j (f_j)
    # source: http://cs231n.github.io/linear-classify/#softmax

    # Compute vector of class scores
    scores = X.dot(W)

    # Compute cross-entropy loss
    # Li = −f_yi + log(∑ e^f_j)

    for i in xrange(num_train):
        # print(np.max(scores[i]), scores[i])
        scores[i] -= np.max(scores[i])
        # print(scores[i])
        exp_sum = np.sum(np.exp(scores[i]))
        loss += -scores[i, y[i]] + np.log(exp_sum)

        # Compute gradient
        # Very good explanation about how to get derivative of softmax function
        # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        # NOTE: for X = (N, D) and W = (D, C)
        # dL/dW_ij = (P_j - (y[i] == j)) * x_i
        for j in xrange(num_classes):
            p_j = np.exp(scores[i, j]) / exp_sum
            dW[:, j] += (p_j - (j == y[i])) * X[i, :]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    # Add regularization to the gradient.
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]

    # Compute scores
    scores = X.dot(W)

    # Use trick with constant to improve numerical stability of computation
    # print(np.max(scores, axis=1).shape)
    # returns (500,) and to be broadcasted it must be reshaped into a column
    # using reshape(-1,1), but keepdims leaves in the result the axes
    # which are reduced as dimensions with size one.
    # print(np.max(scores, axis=1, keepdims=True).shape) will return (500, 1)

    scores -= np.max(scores, axis=1, keepdims=True)

    # Compute −f_yi from Li formula
    f_yi = scores[np.arange(num_train), y]
    # Compute ∑ e^f_j summarize exponents over all classes of each sample
    sum_exp = np.sum(np.exp(scores), axis=1)
    # Compute the loss Li = −f_yi + log(∑ e^f_j)
    loss = np.sum(-f_yi + np.log(sum_exp))

    # Compute the gradient
    p = np.exp(scores) / sum_exp.reshape(-1, 1)  # (N, C) /(N, 1)
    ind = np.zeros_like(p)
    ind[np.arange(num_train), y] = 1  # (N, C)
    dW = X.T.dot(p - ind)  # (D, N)x(N, C)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    # Add regularization to the gradient.
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
