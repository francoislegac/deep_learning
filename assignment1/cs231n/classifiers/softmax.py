from builtins import range
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

    num_train, dim = X.shape
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    scores = X.dot(W) #(N, D) x (D,C) = (N,C)
    for i in np.arange(num_train):
        sum_exp_yj = 0
        syi = scores[i, y[i]]
        for l in np.arange(num_classes):            
            sum_exp_yj += np.exp(scores[i, l])

        for j in np.arange(num_classes):
            if j == y[i]:
                dW[:,j] += (np.exp(syi)/sum_exp_yj-1)*X[i]
            else:
                dW[:,j] += (np.exp(scores[i,j])/sum_exp_yj)*X[i]
        
        loss += - np.log(np.exp(syi)/sum_exp_yj)
    
    loss /= num_train
    loss += reg * np.sum(W*W)
    
    dW /= num_train
    dW += reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W) #(N,C)
    max_sj = np.amax(scores, axis=1)
    scores = scores - max_sj[:,np.newaxis]
    exp_scores = np.exp(scores) 
    sum_exp = exp_scores.sum(axis=1)
    probas = exp_scores/sum_exp[:,np.newaxis] #(N,C)
    probas_yi = probas[np.arange(num_train), np.squeeze(y)]
    probas[np.arange(num_train), np.squeeze(y)] += - 1
    #grad
    dW = X.T.dot(probas) #(D,N) x(N,C) -> (D,C)
    
    Li = - np.log(probas_yi)
    loss = Li.sum(axis=0)

    loss /= num_train
    loss += reg * np.sum(W*W)

    dW /= num_train
    dW += reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
