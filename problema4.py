import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import comb


# Reading the data
req = 'https://raw.githubusercontent.com/Duhart/diagnosis_data/master/media_exposure.csv'
df = pd.read_csv(req)

print(df.shape)
df.head()

print(df['freq'].describe())
plt.hist(df['freq'])
plt.xlabel('Frequency of internet use', fontsize=16)
plt.ylabel('Number of persons', fontsize=16)
plt.show()

print(df['usage'].describe())
plt.hist(df['usage'])
plt.xlabel('Time of internet use', fontsize=16)
plt.ylabel('Number of persons', fontsize=16)
plt.show()

plt.plot(df['freq'], df['usage'], 'bo')
plt.xlabel('Frequency', fontsize=16)
plt.ylabel('Time (usage)', fontsize=16)
plt.show()

#%% Finding the value of c

n_visits = 200 # Number of visites in the month from people in the sample
d=30 # Asumed number of days in the month

# Finding the value of the constant 2
c = np.sum((df['freq']*d*df['usage'])/n_visits)-df['usage'].max()

#%% Classification model

def visita(freq, usange, u_max, thresh, c, d):
    """Function to find if a pearson will visite the cite in a month
    IMPUT
    - freq: (int) Frequency with witch they use the internet
    - usage: (int) Avarage time they use the internet each time
    - u_max: (float) Maximum usage of the internet in the sample
    - thresh: (float) Trheshold to separate visiters 
    - c: (float) Value of the constant
    - d: (int) Number of days in the month
    OUTPUT
    - bol: Indicating if the person is expected to ovisit the cite (1)
        e.o.c. (0)
    - px: Probability of visiting the cite at least one"""
    
    p = usange/(u_max+c)
    n = freq*d
    px = [comb(n, x)*(p**x)*((1-p)**(n-x)) for x in range(1, n)]   
    px = sum(px)
    
    return int(px>thresh), px


""" Finding the expectad pearsosns that will click """    
u_max = df['usage'].max()
thresh = 0

for i in range(df.shape[0]):
    df.loc[i, ['visit', 'proba']] = visita(df.loc[i, 'freq'], df.loc[i, 'usage'], u_max, thresh, c, d)

# Number of people that will or will not visite
df['visit'].value_counts()

# Distribution of the probability to click more than once
plt.hist(df['proba'])
plt.xlabel('Clicks')
plt.ylabel('Number of persons')
plt.show()

#%% Finding the expected amount of clicks from each person

df['freq-usag'] = df['freq']*d*(df['usage']/(df['usage'].max()+c))

print('Comprobando:', np.sum(df['freq-usag']))

fig = plt.figure(figsize=(15,4))
plot = fig.add_subplot(111)
plt.plot(df['ID'], df['freq-usag'], 'bo')
plt.xlabel('ID', fontsize=14)
plt.ylabel('Expected number of visits', fontsize=14)
plot.tick_params(axis='x', labelsize=14)
plot.tick_params(axis='y', labelsize=14)
plt.show()

fig = plt.figure(1)
plot = fig.add_subplot(111)
plt.hist(df['freq-usag'])
plt.xlabel('Expected number of clicks', fontsize=14)
plt.ylabel('Number of persons', fontsize=14)
plot.tick_params(axis='x', labelsize=14)
plot.tick_params(axis='y', labelsize=14)
plt.show()

fig = plt.figure(1)
plot = fig.add_subplot(111)
plt.scatter(df['usage'], df['freq-usag'], c = df['freq'])
plt.xlabel('Usage', fontsize=14)
plt.ylabel('Expected number of visits', fontsize=14)
plot.tick_params(axis='x', labelsize=14)
plot.tick_params(axis='y', labelsize=14)
plt.colorbar()
plt.show()



