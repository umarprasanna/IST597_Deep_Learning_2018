import os  
import tensorflow as tf 
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt  
#from google.colab import files
import io
#uploaded = files.upload()

'''
IST 597: Foundations of Deep Learning
Problem 2: Polynomial Regression & 

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
trial_name = 'p1_fit' # will add a unique sub-string to output of this program
degree = 15 # p, order of model
beta = 1.0 # regularization coefficient
alpha = 0.01 # step size coefficient
eps = 0.00001 # controls convergence criterion
n_epoch = 5000 # number of epochs (full passes through the dataset)
tf.set_random_seed(1618119)
    
# begin simulation
    
def regress(X, theta):
    y_pred = tf.reduce_sum(tf.multiply(X, theta[1]) + theta[0],1)
    return y_pred
    
def gaussian_log_likelihood(mu,y,sigma=1.0):
    	# Function: =  ln(sigma)+ 0.5*noise^2/ 2sigma^2
    noise = regress(mu, theta)-y
    return  0.5 * (np.log(sigma**2) + tf.reduce_sum(tf.square(noise))/(sigma**2))
    	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood
    beta_part = tf.constant(0,tf.float32)
    beta_part = tf.reduce_sum(tf.square(theta[1]))
    cost = gaussian_log_likelihood(X,y, sigma=1.0)/n_sample + (beta_part*beta)/(2*n_sample)  
    return cost
    	
def computeGrad(X, y, theta, beta):
    	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
    dL_dfy = None # derivative w.r.t. to model output units (fy)
    dL_db = 0 # derivative w.r.t. model bias b
    dL_dw = 0 # derivative w.r.t model weights w
    noise = regress(X,theta)-y
    dL_dw = tf.matmul(noise,X) + tf.multiply(beta,theta[1])
    dL_db = tf.reduce_sum(noise,1)
    dL_dw = tf.divide(dL_dw,n_sample)
    dL_db = tf.divide(dL_db,n_sample)
        
    nabla = (dL_db, dL_dw) # nabla represents the full gradient
    return nabla
    
# If using local system
path = os.getcwd() + '/data/prob2.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y'],dtype=np.float32) 
    
# With Google Collab
#data = pd.read_csv(io.StringIO(uploaded['prob2.dat'].decode('utf-8')),header=None,names=['X', 'Y'])
    
    
n_sample = data.shape[0]
# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 
    
# convert from data frames to numpy matrices
X_input = np.array(X.values).astype(np.float32)  
y_input = np.array(y.values).astype(np.float32) 
    
#Input placeholders
X=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
    
# apply feature map to input features x1
X_ = np.zeros((len(X_input), degree))
for i in range(degree):
    for j in range(len(X_input)):
        X_[j][i] = np.power(X_input[j],i+1)
X_input = X_
y_input = np.reshape(y_input,(1,40))
    
#create tensorflow variables w,b and theta as soon above
w = tf.zeros((1,X_input.shape[1]))    
b = tf.zeros(1,1)
theta = (b,w)
        
i = 0
with tf.Session() as sess:
      
    sess.run(tf.global_variables_initializer())
    L = computeCost(X, y, theta,beta=0.0)
    L = sess.run(L,feed_dict={X: X_input,y: y_input})
    print("-1 L = {0}".format(L))
        
    while(i < n_epoch):
          
        #use feeddict to pass variables 
        dL_db, dL_dw = computeGrad(X, y, theta, beta=0.0)
        sess.run([dL_db, dL_dw],feed_dict={X: X_input,y: y_input})
            
        b = theta[0]
        w = theta[1]
            
        # update rules go here...
        w = tf.subtract(w,tf.multiply(alpha,dL_dw))
        b = tf.subtract(b,tf.multiply(alpha,dL_db))
            
        # (note: don't forget to override the theta variable...)
        theta = (b, w)
        theta = sess.run((b,w),feed_dict={X: X_input,y: y_input})
            
        L_i_minus_1 = L
        L = computeCost(X, y, theta, beta=0.0)
        L = sess.run(L,feed_dict={X: X_input,y: y_input})
        L_i = L
        print(" {0} L = {1}".format(i,L))
    	
        if (L_i_minus_1 - L_i < eps):
            break;
        i += 1
            
    # print parameter values found after the search
    print('Theta: (b,w) = ',theta)
    #Save everything into saver object in tensorflow
    writer =tf.summary.FileWriter("output_folder",sess.graph)
    #Visualize using tensorboard
        
    kludge = 0.25
    # visualize the fit against the data
    X_test = np.linspace(data.X.min(), data.X.max(), 100).astype(np.float32)
    X_feat = np.expand_dims(X_test, axis=1) # we need this otherwise, the dimension is missing (turns shape(value,) to shape(value,value))
        
    # apply feature map to input features x1
    X_feat_ = np.zeros((len(X_test), degree)).astype(np.float32)
    for j in range(len(X_feat)):
        for i in range(degree):
            X_feat_[j][i] = np.power(X_test[j],i+1)
    X_feat = X_feat_
        
        
    plt.plot(X_test, sess.run(regress(X, theta),feed_dict={X: X_feat}), label="Model")
    plt.scatter(X_input[:,0], y_input, edgecolor='g', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
    plt.ylim((np.amin(y_input) - kludge, np.amax(y_input) + kludge))
    plt.legend(loc="best")
    #plt.savefig('modelP2_alpha'+str(alpha)+'_beta'+str(beta)+'.png')
    plt.show()
        
        
    writer.close()



