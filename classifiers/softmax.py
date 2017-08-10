import numpy as np
from random import shuffle
from past.builtins import xrange
import math

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
  N,D = X.shape
  C = W.shape[1]
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(N):
    f = X[i].dot(W)
    f -= np.max(f) # numeric statbility
    sum_e_f = np.sum(np.exp(f))
    e_f = np.exp(f)
    P = e_f / sum_e_f
    loss += -math.log(P[y[i]])
    
    for k in range(C):
      dW[:,k] += (P[k] - (y[i] == k)) * X[i,:]
    
  loss /= N
  dW /= N
  loss += reg * np.sum(W * W)
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
  N = X.shape[0]
  C = W.shape[1]
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W) # (N,C)
  f -= np.max(f, axis=1).reshape((N, 1)) # numerical stability
  e_f = np.exp(f) # (N,C)
  sum_e_f = np.sum(e_f, axis=1).reshape((N,1)) # (N,1)
  P = e_f / sum_e_f # (N,C)
  # doesn't have to build a small p matrix, use matrix indexing
  L = np.log(P[np.arange(N), y]) # (N,)
  loss += -np.sum(L)
  loss /= N
  loss += reg * np.sum(W * W)
    
  s_p = np.zeros((N, C)) # (N,C)
  # set some of the items, matrix indexing important 
  s_p[np.arange(N), y] = 1
  dW = -X.T.dot(s_p - P) # 
  dW /= N
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

