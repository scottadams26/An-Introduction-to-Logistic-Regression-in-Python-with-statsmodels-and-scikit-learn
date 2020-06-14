"""
Python code for "An Introduction to Logistic Regression in Python with statsmodels and scikit-learn"
Scott Adams
"""

#=========================#
# Example Data Generation #
#=========================#

from scipy.stats import norm
from scipy.linalg import cholesky
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Below is the code I used to generate the .npz file

# method = 'cholesky'
# n = 100
# r = np.array([
#         [1.95, 1],
#         [1,  1.2],
#     ])
# x = norm.rvs(size=(2, n))
# x = x-x.min()
# c = cholesky(r, lower=True)
# y = np.dot(c, x)
# y = np.where(y > y.mean(), 1, 0)

#np.savez("data/randomData", x[1], y[1])

npzfile = np.load("data/randomData.npz")
x = npzfile['arr_0']
x = x-x.min()
y = npzfile['arr_1']

plt.figure(figsize=(5, 5))
ax = plt.axes()
ax.scatter(x, y, color='b', alpha=0.20)
ax.set_xlabel('x')
ax.set_ylabel('Y')
plt.savefig('img/binaryScatter.png')


#=========================#
# Linear and Logit Models #
#=========================#

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

# Logit Model
prelogit = LogisticRegression(C=1e9)
prelogit.fit(x.reshape(-1, 1), y)
prelogit.intercept_
prelogit.coef_

X_new = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
y_proba = prelogit.predict_proba(X_new)
y_logit = np.log(y_proba/(1-y_proba))
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

# Linear Probability Model
prelm = LinearRegression()
prelm.fit(x.reshape(-1, 1), y)
prelm.intercept_
prelm.coef_


#===============================#
# Example Linear vs Logit Graph #
#===============================#

# Linear Regression
plt.figure(figsize=(5, 5))
ax = plt.axes()
ax.scatter(x, y, color='b', alpha=0.20)
ax.plot(x, prelm.predict(x.reshape(-1, 1)), color='black', alpha=0.70, linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('Y')
ax.set_ylim([-0.4, 1.4])
plt.savefig('img/linearProb.png')

# Logistic Regression
plt.figure(figsize=(5, 5))
ax = plt.axes()
ax.scatter(x, y, color='b', alpha=0.20)
ax.plot(X_new, y_proba[:, 1], linewidth=3, color='black', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('Y')
plt.savefig('img/logitProb.png')


#============#
# Logit By X #
#============#

plt.figure(figsize=(5, 5))
ax = plt.axes()
ax.plot(X_new, y_logit[:, 1], linewidth=3, color='black', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('log-odds(Y=1)')
plt.savefig('img/logitX.png')


#=============#
# Logit Quads #
#=============#

# Logit Plot
plt.figure(figsize=(5, 5))
ax = plt.axes()
ax.plot(X_new, y_proba[:, 1], linewidth=4, color='black', alpha=0.8)

plt.axhline(0.5, color='black', linestyle='dashed', linewidth=0.75)
plt.axvline(decision_boundary, color='black', linestyle='dashed', linewidth=0.75)

plt.plot(
  x[(y==0) & (x < decision_boundary)], y[(y==0) & (x < decision_boundary)],
  "bo",
  alpha=0.3,
  label="Accurate Classification")

plt.plot(
  x[(y==0) & (x >= decision_boundary)], y[(y==0) & (x >= decision_boundary)],
  "x",
  color = "red",
  alpha=0.8,
  label="Inaccurate Classification")

plt.plot(
  x[(y==1) & (x < decision_boundary)], y[(y==1) & (x < decision_boundary)],
  "x",
  color = "red",
  alpha=0.8)

plt.plot(
  x[(y==1) & (x >= decision_boundary)], y[(y==1) & (x >= decision_boundary)],
  "bo",
  alpha=0.3)

ax.set_xlabel('x')
ax.set_ylabel('Pr(Y=1)')
plt.text(4.0, 0.77, "True Positives", fontsize=8, color="k", ha="center")
plt.text(0.7,0.77, "False Negatives", fontsize=8, color="k", ha="center")
plt.text(4.0, 0.2, "False Positives", fontsize=8, color="k", ha="center")
plt.text(0.7,0.2, "True Negatives", fontsize=8, color="k", ha="center")
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.00), shadow=False, ncol=2)
plt.savefig('img/logitQuad.png')


#=========================#
# Load PIMA Diabetes Data #
#=========================#

import pandas as pd

diab = pd.read_csv("data/pima_diabetes.csv")
diab = diab.loc[:, ["Glucose", "Outcome"]]
diab = diab.loc[diab["Glucose"] != 0, :]
diab.head()
diab.describe()

# Min center age for meaningful intercept
diab["Glucose"] = diab["Glucose"] - diab["Glucose"].min()


#======================================#
# Logistic Regression With Statsmodels #
#======================================#

import statsmodels.formula.api as smf

m1 = smf.logit(
  formula='Outcome ~ Glucose',
  data=diab) \
.fit()

m1.summary()
m1.pred_table()

# Calculate log-likelihood manually just for fun
m1p = np.log(m1Probs)
m1q = np.log(1 - m1Probs)
y = np.array(diab["Outcome"])
yinv = 1-y

llpt1 = y*m1p
llpt2 = yinv*m1q
llpt3 = llpt1 + llpt2
ll = np.sum(llpt3)


# Plot Predicted Probs
"""
First, generate a data frame of 1000 values between the min & max Glucose values.
This will allows us to plot predicted probabilities across a set of continuous values.
Then, generate the predicted probabilites and plot them against the new data frame
of glucose values
"""

glucoseNew = pd.DataFrame({'Glucose': np.linspace(diab["Glucose"].min(), diab["Glucose"].max(), 1000)})
predProbs = m1.predict(glucoseNew)

plt.figure(figsize=(5, 5))
ax = plt.axes()
ax.scatter(diab["Glucose"], diab["Outcome"], color='b', alpha=0.20)
ax.scatter(glucoseNew, predProbs , color="black", s=4)
ax.set_xlabel('Glucose - 44')
ax.set_ylabel('Pr(Diabetic)')
plt.savefig('img/diabetesGlucosePreds.png')


#=======================================#
# Logistic Regression With Scikit-Learn #
#=======================================#

from sklearn.metrics import log_loss

skLogit = LogisticRegression()
X = diab["Glucose"].to_numpy().reshape(-1, 1)
m2 = LogisticRegression().fit(X, diab["Outcome"])
m2.intercept_
m2.coef_
log_loss(diab["Outcome"], m2.predict_proba(X), normalize=False)

# Compare to m1 coefficients and log-likelihood
m1.summary()

# Calculte Decision Boundary
glucoseNew = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
glucoseNew[m2.predict_proba(glucoseNew)[:, 1] >= 0.5][0]


# Confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(diab["Outcome"], m2.predict(X)))
m1.pred_table()  # confusion matrix from statsmodels
