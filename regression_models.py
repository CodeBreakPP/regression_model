
## iterations: the number of epochs to optimize the parameters, default is 100
## alpha: the learning rate of the gradient descent
def LinearRegression(X, Y, iterations=100,alpha=0.01):
    ## convert list to numpy array for the convenience of matrix computation
    import numpy as np
    matX=np.asarray(X)
    n,p=matX.shape[0],matX.shape[1]
    y_true=np.asarray(Y)
    ## randomly initialize the beta_hat
    beta_hat=np.random.rand(p,1)
    for _ in range(iterations):
        y_pred=np.dot(matX,beta_hat)
        res=y_pred-y_true
        gradient=np.dot(matX.T,res)
        beta_hat=beta_hat-alpha/n*gradient
    return beta_hat


def LogisticRegression(X, Y, iterations=100,alpha=0.01):
    import numpy as np
    matX=np.asarray(X)
    n,p=matX.shape[0],matX.shape[1]
    y_true=np.asarray(Y)
    beta_hat=np.random.rand(p,1)
    for _ in range(iterations):
        ## sigmoid function
        z=np.dot(matX,beta_hat)
        y_pred=1/(1+np.exp(-z))
        res=y_pred-y_true
        gradient=np.dot(matX.T,res)
        beta_hat=beta_hat-alpha/n*gradient
    return beta_hat
    
