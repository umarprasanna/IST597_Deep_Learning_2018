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
Problem 1a: Softmax Regression \& the XOR Problem

@author - Alexander G. Ororbia II and Ankur Mali
'''

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
	# WRITEME: write your code here to complete the routine
	W = theta[0]
	b = theta[1]
	dW = W * 0.0
	db = b * 0.0	
	return (dW,db)

def computeCost(X,y,theta,reg):
	# WRITEME: write your code here to complete the routine
	return 0.0

def predict(X,theta):
	# WRITEME: write your code here to complete the routine
	W = theta[0]
	b = theta[1]
	# evaluate class scores
	scores = 0.0
	# compute the class probabilities
	probs = 0.0
	return (scores,probs)



np.random.seed(0) #Provide your unique Random seed
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
#Train a Linear Classifier

# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1

# initialize parameters in such a way to play nicely with the gradient-check!
#W = 0.01 * np.random.randn(D,K)
#b = np.zeros((1,K)) + 1.0

initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
W = tf.Variable(initializer([D, K]))
b = tf.Variable(tf.random_normal([K]))
theta = (W,b)

# some hyperparameters
reg = 1e-3 # regularization strength

nabla_n = computeNumGrad(X_tf,Y_tf,theta,reg)
nabla = computeGrad(X_tf,Y_tf,theta,reg)
nabla_n = list(nabla_n)
nabla = list(nabla)
#Initialize your variables
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

# Re-initialize parameters for generic training
initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.float32) #You can use Xavier or Ortho for weight init
#If using other init compare that with Guassian init and report your findings
W = tf.Variable(initializer([D, K]))
b = tf.Variable(tf.random_normal([K]))
theta = (W,b)

#play with hyperparameters for better performance 
n_e = 100 #number of epochs
check = 10 # every so many pass/epochs, print loss/error to terminal
step_size = 1e-0
reg = 0.0 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
with tf.Session as sess(): #You can exclude this ans use sess.run() whenever needed 
	for i in xrange(n_e):
		# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
		loss = 0.0
		if i % check == 0:
			print "iteration %d: loss %f" % (i, loss)

		# perform a parameter update
		# WRITEME: write your update rule(s) here
 
# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
sys.exit(0) 

# evaluate training set accuracy
scores, probs = predict(X_tf,theta)
#scores = np.dot(X, W) + b
predicted_class = sess.run(tf.argmax(scores, axis=1))
print 'training accuracy: %.2f' % sess.run((tf.reduce_mean(predicted_class == y)))
