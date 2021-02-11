#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


arr=np.array([[1,2,3,4]])
s= sum(arr)

print(arr.shape)
print(s)


# In[3]:


from pylab import rcParams
rcParams['figure.figsize'] = 12, 8


# In[4]:


data = pd.read_csv("DMV_Written_Tests.csv")
data.head()


# In[5]:


data.info()


# In[6]:


scores= data[['DMV_Test_1','DMV_Test_2']].values
results= data['Results'].values
scores.shape


# In[7]:


passed= (results==1).reshape(100,1)
failed= (results==0).reshape(100,1)
ax = sns.scatterplot(x = scores[passed[:, 0], 0],
                     y = scores[passed[:, 0], 1],
                     marker = "^",
                     color = "green",
                     s = 60)
sns.scatterplot(x = scores[failed[:, 0], 0],
                y = scores[failed[:, 0], 1],
                marker = "X",
                color = "red",
                s = 60)

ax.set(xlabel="DMV Written Test 1 Scores", ylabel="DMV Written Test 2 Scores")
ax.legend(["Passed", "Failed"])


plt.show()


# In[18]:


#define Sigmoid function

def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[19]:


sigmoid(0)


# In[20]:



def compute_cost(Y,A,m):
    error = (Y * np.log(A)) + ((1 - Y) * np.log(1 - A))
    cost = -(1 / m ) * np.sum(error)
    return cost


# In[22]:


x=scores.T
print("scores dimensions :" ,x.shape)


mean_scores = np.mean(x, axis=1)
std_scores = np.std(x, axis=1)

mean= np.reshape(mean_scores,(2,1))
std=np.reshape(std_scores,(2,1))
print("mean scores dim:", mean.shape)
print("std dim:",std.shape)

X= (x-mean)/std
print("shape of standardized X: ", X.shape)

print(X)
Y= results.reshape(1,100)
print("results dimensions: ", Y.shape)
#results_data

rows = X.shape[0]
                        
#print(rows)

weights=np.zeros((rows,1))
print("weights dimensions:", weights.shape)
print("weights :", weights)
bias=0


# In[23]:


def gradient_Descent(X,Y,weights,bias,alpha):
    
    Z= np.dot(weights.T,X)+bias
    #print("shape of Z:",Z.shape)
    A= sigmoid(Z)
    dZ=A-Y
    #print("shape of dZ:", dZ.shape)
    #print("shape of A:", A.shape)
    m= X.shape[1]
    #print("Number of data points:", m)
    dW = (1 / m) * np.dot(X, dZ.T)
    #print("shape of dW:", dW.shape)
    db = (1 / m) * np.sum(dZ)
    alpha=0.01
    weights= weights- (alpha*dW)
    #print("dim of updated weights:",weights.shape)
    bias= bias-(alpha*db)
    cost= compute_cost(Y,A,m)
    #print(cost)
    #costs.append(cost)
    
    return weights, bias, cost


# In[24]:


alpha=0.7;
iterations=20000;
costs=[]
for i in range(iterations):
    
    weights,bias,cost =gradient_Descent(X,Y,weights,bias,alpha)
    costs.append(cost)
    

print("weights: ",weights)
print("bias",bias)
print(costs)


# In[25]:


import matplotlib.pyplot as plt

plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("$J(\Theta)$")
plt.title("Values of Cost Function over iterations of Gradient Descent");


# In[27]:


x_plot= X.T
print(x_plot.shape)

passed= (results==1).reshape(100,1)
failed= (results==0).reshape(100,1)


sns.scatterplot(x = x_plot[passed[:, 0], 0],
                y = x_plot[passed[:, 0], 1],
                marker = "^",
                color = "green",
                s = 60)
ax = sns.scatterplot(x = x_plot[failed[:, 0], 0],
                    y = x_plot[failed[:, 0], 1],
                    marker = "X",
                    color = "red",
                    s = 60)
ax.legend(["Passed", "Failed"])
ax.set(xlabel="DMV Written Test 1 Scores", ylabel="DMV Written Test 2 Scores")



x_boundary = np.array([np.min(x_plot[:, 1]), np.max(x_plot[:, 1])])
y_boundary = -(bias + weights[0] * x_boundary) / weights[1]

sns.lineplot(x = x_boundary, y = y_boundary, color="blue")
plt.show();


# In[ ]:





# In[ ]:




