import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')
df

%matplotlib inline
plt.xlabel('Gender')
plt.ylabel('BrainWeight')
plt.scatter(df.Gender,df.BrainWeight,color='red',marker='+')

X=df['Gender'].values
Y = df['BrainWeight'].values

# Calculating coefficient

mean_x = np.mean(X)
mean_y = np.mean(Y)

n = len(X)

numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

print("coefficients for regression",b1, b0)

# Plotting Values and Regression Line
%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 5.0)

y = b0 + b1 * X

plt.plot(X, y, color='blue', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='green', marker='*',label='Scatter data')

plt.xlabel('Gender')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

# Calculating R2 Score
ss_tot = 0
ss_res = 0
for i in range(n):
    y_pred = b0 + b1 * X[i]
    ss_tot += (Y[i] - mean_y) ** 2
    ss_res += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)
print("R2 Score",r2)

print(325.57342104944223 + 0.26342933948939945 * 5000)