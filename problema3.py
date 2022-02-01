""" Ejercicio 3"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d    
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from __future__ import division
from sympy import *
from scipy.optimize import curve_fit

from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

""" Importing and visualizing the data """

# Reading the data
req = 'https://raw.githubusercontent.com/Duhart/diagnosis_data/5c188f4b49cab406960b532b8c01cb346c618afa/media_optimisation.csv'
df = pd.read_csv(req)

print(df.shape)
df.head()

# Ploting the data
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(df['tv'].values, df['digital'].values, df['revenue'].values, 'b', alpha=0.5)
ax.set_xlabel('tv')
ax.set_ylabel('digital')
ax.set_zlabel('revenue')
plt.show()

#%% Fitting a function to the data 

# Using non-linear least squares

""" Defining the functions that can will be approximated """

def hiperbolico(data, a, b, c):
    x = data[0]
    y = data[1]
    return  x**2/a**2 - y**2/b**2 + c

def sin_cos(data, a, b, c):
    x = data[0]
    y = data[1]
    return  (c*np.cos(a*x))*(np.sin(b*y))

def experimento(data, a, b, c, d):
    x = data[0]
    y = data[1]
    return  x**2/a**2 - y**2/b**2 + x*y/c + d

def cuadratica(data, a, b, c, d):
    x = data[0]
    y = data[1]
    return a*x**2 + b*y**2 + c*x*y + d

def cuadratica2(data, a, b, c, d, e, f):
    x = data[0]
    y = data[1]
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

def cubica(data, a, b, c, d, e, f, g, h, j, k):
    x = data[0]
    y = data[1]
    return a*x**3 + b*y**3 + c*x**2*y + d*x*y**2 + e*x*y + f*x**2 + g*y**2 + h*x + j*y + k

def grado4(data, a, b, c, d, e, f):
    x = data[0]
    y = data[1]
    return a*x**4 + b*y**4 + c*x**3*y + d*x*y**2 + e*x*y + f

def sigmoide(data, a, b, c, d, e):
    x = data[0]
    y = data[1]
    return d/(np.exp(-a*x-b*y-c)+e)



def metricas(fun, popt, X_val, y_val):
    """ Evaluating a function
    INPUT
    - fun: Function to evaluate.
    - popt: (list or array) 
        Value of the coefficients of the function.
        In the order they appear in fun.
    - X_val, y_val: (list or array)
        coordinates in witch to evaluate the function
    OUTPUT
    - y_val_est: (list) evaluated data
    """
    #### Can be simplified
    
    if len(popt)==3:
        a, b, c = popt
        y_val_est = []
        for i in range(len(y_val)):
            y_val_est.append(fun([X_val.iloc[i]['tv'], X_val.iloc[i]['digital']], a, b, c))
    elif len(popt)==4:
        a, b, c, d = popt
        y_val_est = []
        for i in range(len(y_val)):
            y_val_est.append(fun([X_val.iloc[i]['tv'], X_val.iloc[i]['digital']], a, b, c, d))
    elif len(popt)==5:
        a, b, c, d, e = popt
        y_val_est = []
        for i in range(len(y_val)):
            y_val_est.append(fun([X_val.iloc[i]['tv'], X_val.iloc[i]['digital']], a, b, c, d, e))
    elif len(popt)==6:
        a, b, c, d, e, f = popt
        y_val_est = []
        for i in range(len(y_val)):
            y_val_est.append(fun([X_val.iloc[i]['tv'], X_val.iloc[i]['digital']], a, b, c, d, e, f))
    else:   
        a, b, c, d, e, f, g, h, j, k = popt
        y_val_est = []
        for i in range(len(y_val)):
            y_val_est.append(fun([X_val.iloc[i]['tv'], X_val.iloc[i]['digital']], a, b, c, d, e, f, g, h, j, k))
        
    print('Explainded variance', explained_variance_score(y_val, y_val_est))
    print('Max error', max_error(y_val, y_val_est))
    print('Mean squared error', mean_squared_error(y_val, y_val_est))
    print('R2', r2_score(y_val, y_val_est))
    
    return y_val_est

def evaluando(fun, X_train, y_train, X_val, y_val):
    """ Fitting a function to the data.
        Prints evaluation metrics.
    INPUT
    - fun: function to fit
    - X_val, y_val: (list or array)
        coordinates in witch to evaluate the function
    OUTPUT
    - popt: (list) approximated parameters
    - y_ent_est: (list) trained data evaluated by the fun
    - y_val_est: (list) validation data evaluated by the fun
    """
    # Finding the parameters that adjust the data
    popt, pcov = curve_fit(fun, [X_train['tv'].values, X_train['digital'].values], y_train.values.reshape(-1))
    
    # Metrics
    print('Train')
    y_ent_est = metricas(fun, popt, X_train, y_train)
    print('Test')
    y_val_est = metricas(fun, popt, X_val, y_val)
    
    return popt, y_ent_est, y_val_est


""" Preparing the data
    Separating train, test and validation """

X = df.drop(columns=['revenue']).copy()
y = df[['revenue']].copy()

# Train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


""" Testing the different models """

modelos = [hiperbolico, sin_cos, experimento, cuadratica, cuadratica2, cubica, grado4, sigmoide]

for i in modelos:
    print(i)
    param, y_ent_pred, y_val_pred = evaluando(i, X_train, y_train, X_val, y_val)

""" Best found model """
# Cubic equation 
cub, y_cub_ent, y_cub_val = evaluando(cubica, X_train, y_train, X_val, y_val)

# Metrics for the test set
y_elip_test = metricas(cubica, cub, X_test, y_test)

# Plotting the solution
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_train['tv'].values, X_train['digital'].values, y_cub_ent, c='blue', alpha=0.1, label='Aproximation')
ax.scatter3D(df['tv'].values, df['digital'].values, df['revenue'].values, color='green', alpha=0.1, label='Data points')
ax.set_xlabel('tv')
ax.set_ylabel('digital')
ax.set_zlabel('revenue')
plt.legend()
plt.show()
#plt.save('kantar_savefig.png')

#%% Optimizing the revenue

from sympy import solve, exp


def solucion(cub, B):
    """Solving the linear system
    to find the optimal investment proportions
    INPUT
    - cub: (list) function parameters
    - B: (int or float) total investment
    OUTPUT
    - prop:(list) proportion to invest in TV, digital"""
    a, b, c, d, e, f, g, h, j, k = cub
    x, y, t = symbols('x y t')
    
    # solve system of equations
    sol = solve([3*a*x**2 + 2*c*x*y + d*y**2 + e*y + 2*f*x + h - t, 3*b*y**2 + c*x**2 + 2*d*x*y + e*x + 2*g*y + j - t, x+y-B], x, y, t, dict=True)
    
    # Finding the right proportion 
    if (sol[0][x]>0) & (sol[0][y]>0):
        prop = [sol[0][x]/B, sol[0][y]/B]
    else:
        prop = [sol[1][x]/B, sol[1][y]/B]
    
    print('Proportion of optimal investment in TV:', prop[0])
    return  prop

# Best parameters found
cub = [-3.76261298e-06, -2.43174165e-06, -8.27993593e-09, -6.07666051e-09,
        9.37243038e-06,  6.47849728e-03,  3.33580328e-03, -1.78878582e+00,
       -4.00551002e-02,  7.08276753e+01]

# Solving for each week
S1 = solucion(cub, 200)
S2 = solucion(cub, 800)
S3 = solucion(cub, 1600)