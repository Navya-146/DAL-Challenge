#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


data=pd.read_excel("C:\\Users\\pooja\\OneDrive\\Desktop\\Documents\\IITM\\DAL\\lab 2\\lab 2 dataset.xlsx")


# In[3]:


fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Two Cyber Physical Systems')
fig.set_figwidth(10)
fig.set_figheight(5)

ax1.plot(data.SpringPos, 'r+')
ax1.set_ylabel('Spring Position')
ax2.plot(data.StockPrice, 'b.')
ax2.set_ylabel('Stock Price')
ax2.set_xlabel('time')

plt.show()


# ## TASK 1

# In[4]:


y2 = pd.DataFrame({"x":range(226), "y":data.StockPrice})
yy = np.array(y2.y) 
xx = np.expand_dims(y2.x, 1)


# In[5]:


def estimateBeta(X, y):
    numerator = np.matmul(np.transpose(X), y)
    denom = np.matmul(np.transpose(X), X)
    denom_inv = np.linalg.inv(denom)
    beta = np.matmul(denom_inv, numerator)
    return beta

def predict(beta, X):
    if len(X.shape) != 2:
        X = np.expand_dims(X,1)
    beta = np.array(beta)
    return np.matmul(X, beta)

def SSE(y, yhat):
    return np.sum((y-yhat)**2)


# In[6]:


#fITTING A LINE WITHOUT INTERCEPT

beta=estimateBeta(xx,yy)
yhat1=predict(beta,xx)
sse=SSE(yy,yhat1)
print("beta={},sse={}".format(beta[0],sse))


# In[7]:


plt.title("Regression Line on Stock Price")
plt.plot(y2.x, y2.y, 'r+', label="data")
plt.plot(y2.x, yhat1, 'b-', label="regression line")
plt.ylabel('Stock Price')
plt.xlabel('Time')
plt.legend()
plt.show()


# In[8]:


theta = np.arange(0, 61, 5)

sse_trig = float('inf')
beta_trig = None
theta_best=None

sse_vals=[]
for i in theta:
    test_beta=np.array([np.tan(np.radians(i))])
    err = SSE(yy, predict(test_beta,xx))
    sse_vals.append(err)
    if err < sse_trig: 
        sse_trig = err
        beta_trig = test_beta
        theta_best=i
yhat2=predict(beta_trig,xx)
print("beta={}, sse={}".format(beta_trig,sse_trig))
print("Theta that minimises SSE is=",theta_best,"degrees")


# In[9]:


plt.plot(theta,sse_vals)
plt.ylabel('SSE')
plt.xlabel('Theta')
plt.title("Theta v/s SSE")
plt.show()


# In[10]:


from sklearn.linear_model import LinearRegression

model=LinearRegression(fit_intercept=False)
model.fit(xx,yy)
beta_regmodel=model.coef_[0]
print("beta=",beta_regmodel) 
#Same as OLS


# In[11]:


#comparison
comparison_df=pd.DataFrame([['OLS',beta[0],sse],['Trig_slope',beta_trig[0],sse_trig]
                            ],columns=['model','m','SSE'])
comparison_df


# In[12]:


plt.title("Regression lines")
plt.plot(y2.x, y2.y, 'r+')
plt.plot(y2.x, yhat1, 'b-')
plt.plot(y2.x, yhat2, '-g')
plt.ylabel('Stock Price')
plt.xlabel('Time')
plt.legend(['data_pts','trig_slope','OLS and Sklearn'])
plt.show()


# In[13]:


#with intercept

y2df = pd.DataFrame({"bias":np.ones(226), "x":range(226), "y":data.StockPrice})
yy = np.array(y2df.y) 
xx = np.array(y2df[["bias","x"]])

beta3 = estimateBeta(xx, yy)
print("beta =", beta3)
yhat3 = predict(beta3, xx)
loss = SSE(yy, yhat3)
print("SSE =", loss)


# In[14]:


plt.title("comparison of regression lines: with and without intercept")
plt.plot(y2df.x, y2df.y, 'r+')
plt.plot(y2df.x, yhat1, 'b-')
plt.plot(y2df.x, yhat3, 'g-')
plt.legend(['data','w/o intercept','with intercept'])
plt.ylabel('Stock Price')
plt.xlabel('Time')

plt.show()


# In[15]:


#periodicity
x1 = round(y2.x*beta3[1],2)
x2 = np.sin(x1)

y21 = pd.DataFrame({"bias":np.ones(226),"x":range(226), "x1":x1, "x2":x2, "y":data.StockPrice})
xx = np.array(y21[['bias', 'x1', 'x2']])
yy = np.array(y2.y) 

beta4 = estimateBeta(xx, yy)
print("Beta = ", beta4)
yhat4 = predict(beta4, xx)
loss = SSE(yy, yhat4)
print("SSE = ", loss)


# In[16]:


plt.plot(y2.x, y2.y, 'r+')
plt.plot(y2.x, yhat4, 'b-')
plt.ylabel('Stock Price')
plt.title('Sin curve')
plt.xlabel('Time')
plt.show()


# ### Interpretation: 
# 1. From the initial graph, as the curve of y2 passes from the origin, models without an intercept/ or with 0 intercept should be a better fit. 
# 2. Fitting the closed form OLS solution to y2 is same as using Sklearn LinearRegression model WITHOUT intercept. 
# 3. On analytical comparison between the OLS form solution and m=tan(theta) solution, we can see that OLS is a better model as it has a lesser SSE in comparison to m=tan(theta) model. 
# 4. Visually also, it is observed that the line indicating the OLS form solution passes more points on the graph than the m=tan(theta) solution. 
# 5. Therefore, mathematical model for y2 is: y = 0.118994x + 0
# 6. However, constructing an even better model is possible (in terms of lower SSE) if we incorporate trignometric functions of the 'x' datapoints, as the data in consideration is a sine curve. 

# ## TASK 2

# In[17]:


#interpolation

X=np.array(y21[["x1","x2"]])
y=y21.y

from sklearn.model_selection import train_test_split
X_train,X_test_1,y_train,y_test_1=train_test_split(X,y, random_state=11, test_size=0.4)
X_test,X_eval,y_test,y_eval=train_test_split(X_test_1,y_test_1,random_state=11, test_size=0.5)

model=LinearRegression()
model.fit(X_train,y_train)
beta_interpolation=model.coef_

yhat_eval_interpolation=model.predict(X_eval)
sse_eval_interpolation=SSE(y_eval,yhat_eval_interpolation)
print(sse_eval_interpolation)


# In[18]:


plt.plot(X_eval, y_eval, 'r.')
plt.plot(X_eval, yhat_eval_interpolation, 'b.')
plt.ylabel('Stock Price')
plt.xlabel('Time')
plt.legend(['original data', 'predictions'])
plt.title("Interpolation: Evaluation data")
plt.show()


# In[19]:


yhat_interpolation=model.predict(X_test)
sse_interpolation=SSE(y_test,yhat_interpolation)
print(sse_interpolation)


# In[20]:


plt.plot(X_test, y_test, 'r.')
plt.plot(X_test, yhat_interpolation, 'b.')
plt.ylabel('Stock Price')
plt.xlabel('Time')
plt.legend(['original data', 'predictions'])
plt.title("Interpolation: Test Data")
plt.show()


# In[21]:


#extrapolation

X_train=X[:int(len(X)*0.6)]
y_train=y[:int(len(y)*0.6)]

X_test=X[int(len(X)*0.6):int(len(X)*0.8)]
y_test=y[int(len(y)*0.6):int(len(y)*0.8)]

X_eval=X[int(len(X)*0.8):]
y_eval=y[int(len(y)*0.8):]

model=LinearRegression()
model.fit(X_train,y_train)
beta_extrapolation=model.coef_

yhat_eval_extrapolation=model.predict(X_eval)
sse_eval_extrapolation=SSE(y_eval,yhat_eval_extrapolation)
print(sse_eval_extrapolation)


# In[22]:


plt.plot(X_eval, y_eval, 'r.')
plt.plot(X_eval, yhat_eval_extrapolation, 'b.')
plt.ylabel('Stock Price')
plt.xlabel('Time')
plt.legend(['original data', 'predictions'])
plt.title("Extrapolation: Evaluation data")
plt.show()


# In[23]:


yhat_extrapolation=model.predict(X_test)
sse_extrapolation=SSE(y_test,yhat_interpolation)
sse_interpolation


# In[24]:


plt.plot(X_test, y_test, 'r.', label='original data')
plt.plot(X_test, yhat_extrapolation, 'b.', label='predictions')
plt.ylabel('Stock Price')
plt.xlabel('Time')
plt.legend()
plt.title("Extrapolation: Test data")
plt.show()


# ## TASK 3

# In[25]:


y1 = pd.DataFrame({"x":range(226), "y":data.SpringPos})
yy_ = np.array(y1.y) 
xx_ = np.expand_dims(y1.x, 1)


# In[26]:


beta_=estimateBeta(xx_,yy_)
yhat1_=predict(beta_,xx_)
sse_=SSE(yy_,yhat1_)
print("beta={},sse={}".format(beta_[0],sse_))


# In[27]:


plt.title("Regression line: Spring Position")
plt.plot(y1.x, y1.y, 'r+')
plt.plot(y1.x, yhat1_, 'b-')
plt.ylabel('Spring Position')
plt.xlabel('Time')
plt.show()


# ### Interpretation:
# 1. From the graph, we can see that the curve is sinusoidal with an equal time period of oscillation. The curve spans over 3.5 oscillations, which correspond to 7pi units of the sine function. Therefore, our data can be appropriately scaled to 7pi/226.
# 
# 2. This is a sine surve, so the equation should be of the form y=A(sin(wt))+c.
# 
# 3. From the graph, we can see that this is a damped ocillation, that is, the amplitude of the curve decreases w.r.t time. Hence, we need to incorporate an exponential decay function of scaled x to model the data. That is, A=m*e^(-xk)
# 
# 
# 

# In[28]:


import math

y1df = pd.DataFrame({"x":range(226), "y":data.SpringPos})
yy = np.array(y1df.y) 

x1=y1df.x*math.pi*7/226
x2=np.sin(x1)
x3=np.exp(-1*x1/12.5)  #arbitrary choice
x4=x2*x3
bias=np.ones(226)

y11 = pd.DataFrame({"bias":bias, "x":range(226), "x4":x4, "y":data.SpringPos})
xx = np.array(y11[['bias', 'x4']])
yy = np.array(y11.y)
 
    
model2=LinearRegression()
model2.fit(xx,yy)
yhat3_=model2.predict(xx)
sse=SSE(yy,yhat3_)
print(sse)    
   


# In[29]:


plt.title("Regression line")
plt.plot(y11.x,y11.y,'r+', label="data")
plt.plot(y11.x,yhat3_,'b-', label="regression line")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Spring Position")
plt.show()


# ### Interpretation:
# This curve appropriately fits the data with a low sse. So, it is an okay choice for our model.

# In[30]:


#interpolation

X=np.array(y11[['x4']])
y=np.array(y11.y)

from sklearn.model_selection import train_test_split
X_train,X_test_1,y_train,y_test_1=train_test_split(X,y, random_state=1030, test_size=0.4)
X_test,X_eval,y_test,y_eval=train_test_split(X_test_1,y_test_1,random_state=1030, test_size=0.5)

model3=LinearRegression()
model3.fit(X_train,y_train)
beta_interpolation=model3.coef_

yhat_eval_interpolation=model3.predict(X_eval)
sse_eval_interpolation=SSE(y_eval,yhat_eval_interpolation)
print(sse_eval_interpolation)


# In[31]:


plt.plot(X_eval, y_eval, 'r.')
plt.plot(X_eval, yhat_eval_interpolation, 'b+')
plt.ylabel('Spring Position')
plt.legend(['original data', 'predictions'])
plt.title("Interpolation: Evaluation Data")
plt.show()


# In[32]:


yhat_interpolation=model3.predict(X_test)
sse_interpolation=SSE(y_test,yhat_interpolation)
print(sse_interpolation)


# In[33]:


plt.plot(X_test, y_test, 'r.')
plt.plot(X_test, yhat_interpolation, 'b.')
plt.ylabel('Stock Price')
plt.legend(['original data', 'predictions'])
plt.title("Interpolation: Test Data")
plt.show()


# In[34]:


#extrapolation

X_train=X[:int(len(X)*0.6)]
y_train=y[:int(len(y)*0.6)]

X_test=X[int(len(X)*0.6):int(len(X)*0.8)]
y_test=y[int(len(y)*0.6):int(len(y)*0.8)]

X_eval=X[int(len(X)*0.8):]
y_eval=y[int(len(y)*0.8):]

model=LinearRegression()
model.fit(X_train,y_train)
beta_extrapolation=model.coef_

yhat_eval_extrapolation=model.predict(X_eval)
sse_eval_extrapolation=SSE(y_eval,yhat_eval_extrapolation)
print(sse_eval_extrapolation)


# In[35]:


plt.plot(X_eval, y_eval, 'r.')
plt.plot(X_eval, yhat_eval_extrapolation, 'b.')
plt.ylabel('Spring position')

plt.legend(['original data', 'predictions'])
plt.title("Extrapolation: Evaluation data")
plt.show()


# In[36]:


yhat_extrapolation=model.predict(X_test)
sse_extrapolation=SSE(y_test,yhat_interpolation)
print(sse_interpolation)

plt.plot(X_test, y_test, 'r.', label='original data')
plt.plot(X_test, yhat_extrapolation, 'b.', label='predictions')
plt.ylabel('Stock Price')

plt.legend()
plt.title("Extrapolation: Test data")
plt.show()


# In[ ]:




