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
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    exp_sum = np.sum(np.exp(scores))
    correct = 0.0;
    for j in xrange(num_classes):
      exp_value = np.exp(scores[j])
    
      if j == y[i]:
        dW[:, j] -= X[i].T
        correct = exp_value
      dW[:, j] += exp_value / exp_sum * X[i].T
    
    correct_probability = correct / exp_sum
    loss += -np.log(correct_probability)
    
  loss /= num_train
  loss += reg * np.sum(W * W)
    
  dW /= num_train
  dW += 2 * reg * W
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
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores)
  correct_scores = scores[np.arange(num_train), y]
  exp_scores = np.exp(scores)
  row_exp_scores_sum = np.sum(exp_scores, axis=1)
  row_log_scores_sum = np.log(row_exp_scores_sum)
  loss = np.sum(row_log_scores_sum - correct_scores) / num_train
  loss += reg * np.sum(W * W)
   
  row_to_col = np.reshape(row_exp_scores_sum, (row_exp_scores_sum.shape[0], -1))
  coefficient_matrix = exp_scores / row_to_col
  coefficient_matrix[np.arange(num_train), y] -= 1
  dW = np.dot(X.T, coefficient_matrix) / num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

