from sklearn import datasets
import numpy as np
iris=datasets.load_iris()
x=iris.data[:, :]
y=iris.target
#feature scaling
min_value=np.amin(x,axis=0)
max_value=np.amax(x,axis=0)
for j in range(0,4):
 for i in range(0,150):
  x[i,j]=(x[i,j]-min_value[j])/(max_value[j]-min_value[j])
#converting categorical values into different columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])
y=onehotencoder.fit_transform(y.reshape(-1,1)).toarray()
#distributing the model in training and test set
x_train=x[0:30,:]
y_train=y[0:30]
x_train=np.append(x_train,x[50:80,:],axis=0)
x_train=np.append(x_train,x[100:130,:],axis=0)
y_train=np.append(y_train,y[50:80],axis=0)
y_train=np.append(y_train,y[100:130],axis=0)
x_test=x[30:50,:]
x_test=np.append(x_test,x[80:100],axis=0)
x_test=np.append(x_test,x[130:150],axis=0)
y_test=y[30:50]
y_test=np.append(y_test,y[80:100],axis=0)
y_test=np.append(y_test,y[130:150],axis=0)
ep=[]
cos=[]
#building the model
np.random.seed(0)
w1=np.random.rand(4,3)
b=np.random.rand(1,3)
for epoch in range(0,1000):
 ep=ep+[range]
 z=np.dot(x_train,w1)+b
 def sigmoid(n):
  return(1/(1+np.exp(-n)))
 def sig_deriv(x):
   return (sigmoid(x)*(1-sigmoid(x)))
 y_pred=sigmoid(z)
 error=y_pred-y_train
 print(np.sum(0.5*(error**2))/90)
 cos=cos+[np.sum(0.5*(error**2))]
 #backpropagation algorithm
 dcost_dpred=error
 dpred_dz=sig_deriv(z)
 dcost_dz=dcost_dpred*dpred_dz
 w1-=0.05*np.dot(x_train.T,dcost_dz)
 for num in dcost_dz:
    b-=0.05*num
z2=np.dot(x_test,w1)+b
y_pred2=sigmoid(z2)
error_test=y_pred2-y_test
print(np.sum(0.5*(error_test**2)))      
    