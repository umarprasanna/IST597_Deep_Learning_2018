import os 
import numpy as np
import tensorflow as tf
import pandas as pd  
import matplotlib.pyplot as plt  
import sys
#You have freedom of using eager execution in tensorflow
#Instead of using With tf.Session() as sess you can use sess.run() whenever needed

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Problem 1c: MLPs \& the XOR Problem

@author - Alexander G. Ororbia II and Ankur Mali
'''

def computeCost(X,y,theta,reg):
	# WRITEME: write your code here to complete the routine
	return 0.0

def computeNumGrad(X,y,theta,reg): # returns approximate nabla
	# WRITEME: write your code here to complete the routine
	eps = 1e-5
	theta_list = list(theta)
	nabla_n = []
	# NOTE: you do not have to use any of the code here in your implementation...
	ii = 0
	for param in theta_list:
		param_grad = param * 0.0
		nabla_n.append(param_grad)
		ii += 1
	return tuple(nabla_n)			
			
def computeGrad(X,y,theta,reg): # returns nabla
	W = theta[0]
	b = theta[1]
	W2 = theta[2]
	b2 = theta[3]
	# WRITEME: write your code here to complete the routine
	dW = W * 0.0
	db = b * 0.0
	dW2 = W2 * 0.0
	db2 = b2 * 0.0
	return (dW,db,dW2,db2)

def predict(X,theta):
	# WRITEME: write your code here to complete the routine
	scores = 0.0
	probs = 0.0
	return (scores,probs)
	
np.random.seed(0)
# Load in the data from disk
path = os.getcwd() + '/data/xor.dat'  
data = pd.read_csv(path, header=None) 

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
y = y.flatten()

X_tf = tf.constant(X)
Y_tf = tf.constant(y)
# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1

# initialize parameters in such a way to play nicely with the gradient-check! 
h = 6 #100 # size of hidden layer

initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
W = tf.Variable(initializer([D, h]))
b = tf.Variable(tf.random_normal([h]))
W2 = tf.Variable(initializer([h, K]))
b2 = tf.Variable(tf.random_normal([K]))
theta = (W,b,W2,b2) 

# some hyperparameters
reg = 1e-3 # regularization strength

nabla_n = computeNumGrad(X_tf,Y_tf,theta,reg)
nabla = computeGrad(X_tf,Y_tf,theta,reg)
nabla_n = list(nabla_n)
nabla = list(nabla)

sess = tf.Session()
for jj in range(0,len(nabla)):
	is_incorrect = 0 # set to false
	grad = nabla[jj]
	grad_n = nabla_n[jj]
	grad_sub = tf.subtract(grad_n,grad)
	grad_add = tf.add(grad_n,grad)
	err = tf.div(tf.norm(grad_sub) , (tf.norm(grad_add)))
	if(err > 1e-8):
		print("Param {0} is WRONG, error = {1}".format(jj, sess.run(err)))
	else:
		print("Param {0} is CORRECT, error = {1}".format(jj, sess.run(err)))
# re-init parameters
h = 6 #100 # size of hidden layer

initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
W = tf.Variable(initializer([D, h]))
b = tf.Variable(tf.random_normal([h]))
W2 = tf.Variable(initializer([h, K]))
b2 = tf.Variable(tf.random_normal([K]))
theta = (W,b,W2,b2)

# some hyperparameters
n_e = 100
check = 10 # every so many pass/epochs, print loss/error to terminal
step_size = 1e-0
reg = 0.0 # regularization strength
	
# gradient descent loop
with tf.Session as sess:
	for i in xrange(n_e):
		# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
		loss = 0.0
		if i % check == 0:
			print "iteration %d: loss %f" % (i, loss)

		# perform a parameter update
		# WRITEME: write your update rule(s) here
 
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
sys.exit(0) 

scores, probs = predict(X,theta)
predicted_class = sess.run(tf.argmax(scores, axis=1))
print 'training accuracy: %.2f' % sess.run((tf.reduce_mean(predicted_class == y)))
