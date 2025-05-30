# %%

#dimension reduction using PCA expt-1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load and standardize data
iris = load_iris()
X = StandardScaler().fit_transform(iris.data)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create DataFrame with PCA results
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['target'] = iris.target
df['target_name'] = [iris.target_names[i] for i in iris.target]

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(df['PC1'], df['PC2'], c=df['target'], cmap='Set1', s=60)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset")
plt.grid(True)
plt.tight_layout()
plt.show()

# Explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", np.sum(pca.explained_variance_ratio_))


# %%
#KNN Classification expt-2
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df.info()
iris_df.head()

# Train k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
accuracy=np.mean(y_pred==y_test)

print("Accuracy:", knn.score(X_test, y_test))
for actual, pred in zip(y_test, y_pred):
    print(f"Actual: {actual} ({iris.target_names[actual]}), Predicted: {pred} ({iris.target_names[pred]})")


# %%
# Local Weighted Regression (LWR) expt-3

import numpy as np, pandas as pd, matplotlib.pyplot as plt

def lwr(X, Y, tau, x):  
    w = np.exp(-np.sum((X - x)**2, axis=1) / (2 * tau**2))
    Xb = np.c_[np.ones(X.shape[0]), X]
    W = np.diag(w)
    return np.r_[1, x] @ np.linalg.pinv(Xb.T @ W @ Xb) @ Xb.T @ W @ Y

d = pd.read_csv('housing.csv')
X, Y = d[['housing_median_age']].values, d['median_house_value'].values
xq = np.array([10, 20, 30, 40, 50])
tau = 10
preds = [lwr(X, Y, tau, [x]) for x in xq]

for x, y in zip(xq, preds):
    print(f"Age {x}: {y:.2f}")

# Plot
plt.scatter(X, Y, c='blue', alpha=0.5, label='Data')
plt.scatter(xq, preds, c='red', label='Predictions')
plt.xlabel('Housing Median Age')
plt.ylabel('Median House Value')
plt.legend()
plt.show()


# %%
#linear regression expt-4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv('testdata.csv')
X = df.iloc[:, 1:2].values  # Temperature
y = df.iloc[:, 2].values    # Pressure

# Train model
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# Plot
plt.scatter(X, y, color='red', label='Actual')
plt.scatter(X, predictions, color='blue', label='Predicted')
plt.title('Temperature vs Pressure')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Results
print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"MAE: {mean_absolute_error(y, predictions):.4f}")

# Predict for a new temperature
temp = np.array([[30]])
pred = model.predict(temp)
manual_pred = model.coef_[0] * temp + model.intercept_
print(f"Predicted pressure at {temp[0][0]}°C: {pred[0]:.4f}")
print(f"Manual calculation: {manual_pred[0][0]:.4f}")


# %%
#polynomial regression expt-5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv('testdata.csv')
df.head()

temperature = df.iloc[:, 1:2].values #Input 
preasure = df.iloc[:, 2].values #output
polyreg = PolynomialFeatures(degree=3)
X_poly = polyreg.fit_transform(temperature) 
linreg = LinearRegression()
linreg.fit(X_poly, preasure) 

predicted_preasure = linreg.predict(X_poly)

#Visualising the results
plt.scatter(temperature, preasure, color='red')
plt.scatter(temperature, predicted_preasure, color='blue')
plt.title('Temperature vs Preasure')
plt.xlabel('Temperature')
plt.ylabel('Preasure')
plt.legend(['Real Data', 'Predicted'])
plt.grid()
plt.show()


# %%
#classification using ID3 dataset expt-6
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
dataset = pd.read_csv('zoo.csv') #Consists of 100 entries. 
dataset.info() 

dataset = dataset=dataset.drop('animal_name',axis=1)  
percentage_of_data = 80
train_features = dataset.iloc[:percentage_of_data,:-1]  
train_labels = dataset.iloc[:percentage_of_data,-1]

test_features = dataset.iloc[percentage_of_data:,:-1] 
test_labels = dataset.iloc[percentage_of_data:,-1]

#training the model
tree_model = DecisionTreeClassifier(criterion = 'entropy')  
fit_tree_model = tree_model.fit(train_features,train_labels) 
prediction = fit_tree_model.predict(test_features) 
print("The predicted labels are: ", prediction)

accuracy = (prediction == test_labels).sum() / len(test_labels) 
print("The accuracy of the model is: ", round(accuracy*100,2), "%") 


# %%
#naive bayes classification expt-7
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y, images = data.data, data.target, data.images

# Display first 10 images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f"Class: {y[i]}")
    ax.axis('off')
plt.suptitle("Olivetti Faces - First 10 Samples")
plt.tight_layout()
plt.show()

# Split data & apply PCA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_pca = PCA(n_components=100, whiten=True).fit_transform(X_train)
X_test_pca = PCA(n_components=100, whiten=True).fit(X_train).transform(X_test)

# Train & evaluate model
model = GaussianNB().fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)

print(f"\nAccuracy: {model.score(X_test_pca, y_test):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=np.arange(1, 41), yticklabels=np.arange(1, 41))
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
plt.tight_layout(); plt.show()


# %%



