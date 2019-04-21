import os  
import tensorflow as tf 
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
#from google.colab import files
import io

'''
IST 597: Foundations of Deep Learning
Problem 3: Multivariate Regression & Classification
    
@author - Alexander G. Ororbia II and Ankur Mali
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
        
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
        
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
    
# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
#Change variables to tf.constant or tf.Variable whenever needed
trial_name = 'p6_reg0' # will add a unique sub-string to output of this program
degree = 6 # p, degree of model (LEAVE THIS FIXED TO p = 6 FOR THIS PROBLEM)
beta = 0.0 # regularization coefficient
alpha = 0.1 # step size coefficient
n_epoch = 1000 # number of epochs (full passes through the dataset)
eps = 0.0 # controls convergence criterion
tf.set_random_seed(1618119)
    
# begin simulation
    
def sigmoid(z):
    sig = 0
    sig = tf.divide(1.0,(1.0 + tf.exp(-z)))
    return sig
    
def predict(X, theta):   
    y_ = regress(X, theta)
    y_pred = tf.greater_equal(y_,0.5)
    return y_pred
    	
def regress(X, theta):
    y__ = tf.reduce_sum(tf.multiply(X, theta[1]) + theta[0],1)
    y__ = sigmoid(y__)
    return y__
    
def bernoulli_log_likelihood(p, y):
    y_pred_ = regress(p, theta)
    return tf.reduce_sum(tf.multiply(-y,tf.log(y_pred_))-tf.multiply((1.0-y),tf.log(1.0-y_pred_)))
    	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood
    beta_part = tf.constant(0,tf.float32)
    beta_part = tf.reduce_sum(tf.square(theta[1]))
    loss = (bernoulli_log_likelihood(X,y)/(n_sample)) +  ((beta_part*beta)/(2*n_sample))
    return loss
    	
def computeGrad(X, y, theta, beta): 
    # NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
    dL_dfy = None # derivative w.r.t. to model output units (fy)
    dL_db = 0 # derivative w.r.t. model bias b
    dL_dw = 0 # derivative w.r.t model weights w
    noise = regress(X,theta)-y
    dL_dw = tf.matmul(noise,X)
    dL_db = tf.reduce_sum(noise,1)
    dL_dw = tf.divide(dL_dw,n_sample) + tf.multiply(beta,theta[1])
    dL_db = tf.divide(dL_db,n_sample)
    nabla = (dL_db, dL_dw) # nabla represents the full gradient
    return nabla
    	
path = os.getcwd() + '/data/prob3.dat'  
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
    
# uploaded = files.upload()
# data2 = pd.read_csv(io.StringIO(uploaded['prob3.dat'].decode('utf-8')), header=None, names=['Test 1', 'Test 2', 'Accepted'])
    
    
# positive = data2[data2['Accepted'].isin([1])]  
# negative = data2[data2['Accepted'].isin([0])]
    
n_sample = data2.shape[0]
#Convert positive and negative samples into tf.Variable 
x1 = data2['Test 1']  
x2 = data2['Test 2']
#Convert x1 and x2 to tensorflow variables
# apply feature map to input features x1 and x2
cnt = 0
for i in range(1, degree+1):  
    for j in range(0, i+1):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
        cnt += 1
    
data2.drop('Test 1', axis=1, inplace=True)  
data2.drop('Test 2', axis=1, inplace=True)
    
# set X and y
cols = data2.shape[1]  
X2 = data2.iloc[:,1:cols]  
y2 = data2.iloc[:,0:1]
    
# Placeholder for inputs
X=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
    
# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values).astype(np.float32)    
y2 = np.array(y2.values).astype(np.float32)    
y2 = np.reshape(y2,(1,118))
    
#Convert all numpy variables into tensorflow variables
w = tf.zeros((1,X2.shape[1]))    
b = tf.zeros(1,1)
theta = (b,w)
    
i = 0
#Initialize graph and all variables
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    L = computeCost(X2, y2, theta,beta)
    L = sess.run(L,feed_dict={X: X2,y: y2})
    print("-1 L = {0}".format(L))
    halt=0
        
    while(i < n_epoch and halt == 0):
            
        dL_db, dL_dw = computeGrad(X2, y2, theta, beta)
        #use feeddict to pass variables to pass holder
        sess.run([dL_db, dL_dw],feed_dict={X: X2,y: y2})
        b = theta[0]
        w = theta[1]
        # update rules go here...
        w = tf.subtract(w,tf.multiply(alpha,dL_dw))
        b = tf.subtract(b,tf.multiply(alpha,dL_db))
            
        theta = (b, w)
        theta = sess.run((b,w),feed_dict={X: X2,y: y2})
            
        L_i_minus_1 = L
        L = computeCost(X2, y2, theta, beta)
        L = sess.run(L,feed_dict={X: X2,y: y2})
        L_i = L
    	
        #if (L_i_minus_1 - L_i < eps):
            #halt=1
        	
        print(" {0} L = {1}".format(i,L))
        i += 1
            
    # print parameter values found after the search
    print('Theta: (b,w) = ',theta)
        
    #Save everything into saver object in tensorflow
    writer =tf.summary.FileWriter("output_folder",sess.graph)
    #Visualize using tensorboard
    
    predictions = predict(X, theta)
    sess.run(predictions,feed_dict={X: X2,y: y2})
    # compute error (100 - accuracy)
    err = 0.0
    acc = 0.0
        
    correct = tf.equal(predictions, tf.equal(y,1.0))
    accuracy = tf.reduce_mean( tf.cast(correct, tf.float32) )
    sess.run(accuracy,feed_dict={X: X2,y: y2})
    err = 1 - accuracy
    print('Error = {0}%'.format(sess.run(err*100,feed_dict={X: X2,y: y2})))
        
    # make contour plot
    xx, yy = np.mgrid[-1.2:1.2:.01, -1.2:1.2:.01]
    xx1 = xx.ravel()
    yy1 = yy.ravel()
    grid = np.c_[xx1, yy1]
    grid_nl = []
    # re-apply feature map to inputs x1 & x2
    
    for i in range(1, degree+1):  
      for j in range(0, i+1):
          feat = np.power(xx1, i-j) * np.power(yy1, j)
          if (len(grid_nl) > 0):
             grid_nl = np.c_[grid_nl, feat]
          else:
             grid_nl = feat
    probs = tf.reshape(regress(X, theta),xx.shape)
    sess.run(probs,feed_dict={X:grid_nl})
        
    
    f, ax = plt.subplots(figsize=(8, 6))
    ax.contour(xx, yy, sess.run(probs,feed_dict={X:grid_nl}), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
        
    y2_ = np.reshape(y2,(118))   #Changing the shape back to original
    ax.scatter(x1, x2, c=y2_, s=50,
                   cmap="RdBu", vmin=-.2, vmax=1.2,
                   edgecolor="white", linewidth=1)
    
    ax.set(aspect="equal",
               xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
               xlabel="$X_1$", ylabel="$X_2$")
        
    plt.savefig('modelP3_fit_beta'+str(beta)+'.png')
    
    plt.show()
            
    writer.close()
    

