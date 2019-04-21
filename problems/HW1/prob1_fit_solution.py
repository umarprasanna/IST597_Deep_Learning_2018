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
Problem 1: Univariate Regression

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
alpha = 0.0001 # step size coefficient
eps = 0.00 # controls convergence criterion
n_epoch = 500 # number of epochs (full passes through the dataset)
tf.set_random_seed(1618119)

# begin simulation

def regress(X, theta):
    y_pred = tf.add(tf.multiply(X, theta[1]), theta[0])
    return y_pred

def gaussian_log_likelihood(mu,y,sigma=1.0):
    # Function: =  ln(sigma)+ 0.5*noise^2/ 2sigma^2
    noise = (regress(mu, theta)-y)
    return  0.5 * (np.log(sigma**2) + tf.reduce_sum(tf.square(noise))/(sigma**2))
	
def computeCost(X, y, theta): # loss is now Bernoulli cross-entropy/log likelihood
    cost = gaussian_log_likelihood(X,y, sigma=1.0)/n_sample
    return cost
	
def computeGrad(X, y, theta): 
    dL_dfy = None # derivative w.r.t. to model output units (fy)
    dL_db = 0 # derivative w.r.t. model bias b
    dL_dw = 0 # derivative w.r.t model weights w
    noise = regress(X,theta)-y
    dL_dw = tf.reduce_sum(tf.multiply(noise,X))
    dL_db = tf.reduce_sum(noise)
    dL_dw = tf.divide(dL_dw,n_sample)
    dL_db = tf.divide(dL_db,n_sample)
    
    nabla = (dL_db, dL_dw) # nabla represents the full gradient
    return nabla


# If using local system
path = os.getcwd() + '/data/prob1.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y'],dtype=np.float32) 

# With Google Collab
#data = pd.read_csv(io.StringIO(uploaded['prob1.dat'].decode('utf-8')),header=None,names=['X', 'Y'])

# Basic description of dataset
print(data.describe())
#Scatterplot of dataset
data.plot.scatter(x='X',y='Y',s=20)
plt.title('Scatter plot of the data')
plt.savefig('scatter_plot_data.png')
plt.show()

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

n_sample = data.shape[0]
# convert from data frames to numpy matrices
X_input = np.array(X.values).astype(np.float32)  
y_input = np.array(y.values).astype(np.float32) 


#Placeholder variable for X(input) and Y(output)
X=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)


w = tf.zeros((1,X_input.shape[1]))    
b = tf.zeros(1,1)
theta = (b,w)


i = 0
cost = [] # you can use this list variable to help you create the loss versus epoch plot at the end (if you want)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    L = computeCost(X, y, theta)
    L = sess.run(L,feed_dict={X: X_input,y: y_input})
    print("-1 L = {0}".format(np.round(L,3)))
    L_best = L
    while(i < n_epoch):
        
        
        dL_db, dL_dw = computeGrad(X, y, theta)
        
        #use feeddict to pass variables 
        sess.run([dL_db, dL_dw],feed_dict={X: X_input,y: y_input})
            
        b = theta[0]
        w = theta[1]
        
        # update rules go here...
        w = tf.subtract(w,tf.multiply(alpha,dL_dw))
        b = tf.subtract(b,tf.multiply(alpha,dL_db))
        
        # (note: don't forget to override the theta variable...)
        theta = (b, w)
        theta = sess.run((b,w),feed_dict={X: X_input,y: y_input})
        
        L = computeCost(X, y, theta) # track our loss after performing a single step 
        L = sess.run(L,feed_dict={X: X_input,y: y_input})
        cost.append(L)
        print(" {0} L = {1}".format(i,np.round(L,3)))
        i += 1
        
    #print parameter values found after the search
    print('Theta: (b,w) = ',theta)
    
    #Save everything into saver object in tensorflow
    writer =tf.summary.FileWriter("output_folder",sess.graph)
    #Visualize using tensorboard
    kludge = 0.25 # helps with printing the plots (you can tweak this value if you like)
    # visualize the fit against the data
    X_test = np.linspace(data.X.min(), data.X.max(), 100).astype(np.float32)
    X_test = np.expand_dims(X_test, axis=1)
    
    
    plt.plot(X_test, sess.run(regress(X_test, theta)), label="Model")
    plt.scatter(X_input[:,0], y_input, edgecolor='g', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
    plt.ylim((np.amin(y_input) - kludge, np.amax(y_input) + kludge))
    plt.legend(loc="best")
    plt.title('Fitted linear model against the data samples')
    plt.savefig('model.png')
    plt.show() # convenience command to force plots to pop up on desktop
    writer.close()
    
# visualize the loss as a function of passes through the dataset

x_tick = np.arange(0, len(cost))
plt.plot(x_tick, cost)
plt.xticks(np.arange(min(x_tick), max(x_tick)+1,2))
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.title('Loss by epochs')
plt.savefig('cost_by_epochs.png')
plt.show()

writer.close()
    