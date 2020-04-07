import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('Yelp_cleaned_data.csv')
x = data.iloc[:, 0:23].values#input
y = data['Faker'].values#target
print('researching important feature based on %i total features\n' % x.shape[1])#total no of features

from sklearn.ensemble import ExtraTreesClassifier#applying extra trees classifiers algorithm
fsel = ExtraTreesClassifier().fit(x, y)
from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(fsel, prefit = True)# it selects the best decision trees 
x_new = model.transform(x)
nb_features = x_new.shape[1]
from sklearn.model_selection import train_test_split# training and testing the data
x_train, x_test, y_train, y_test =train_test_split(x_new, y, test_size = 0.2)
features = []
print('%i features identified as important:' % nb_features) #printing the number of selected important features

indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]#sorting them in order
for f in range(nb_features):
    print("%d. feature %s (%f)" % (f + 1, data.columns[2+indices[f]], fsel.feature_importances_[indices[f]]))
    #printing the selected important features
    
    
for f in sorted(np.argsort(fsel.feature_importances_)[::-1][:nb_features]):
    features.append(data.columns[2+f])#storing this features in a list
from sklearn.ensemble import RandomForestClassifier# applying random forest classifier algorithm
algo = RandomForestClassifier(n_estimators = 50)
clf =algo
clf.fit(x_train, y_train)

res = clf.predict(x_test)

from sklearn.metrics import accuracy_score#calculating the accuracy score
acc = accuracy_score(y_test, res)

from sklearn.metrics import confusion_matrix#calculating the confusion matrix
cm = confusion_matrix(y_test, res)

from sklearn.metrics import classification_report#calculating the classification report
cr = classification_report(y_test, res)
print('accuracy score=',acc)
print('confusion_matrix\n',cm)
print('classification_report\n',cr)
#data visualization

dataset = pd.read_csv('Yelp_cleaned_data.csv')#Reading the csv file
X = dataset.iloc[:, [1,2]].values#input values
y = dataset.iloc[:, 45].values#target values


from sklearn.model_selection import train_test_split#training and testing the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler#standardising the training and testing data to filling the missing values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.ensemble import RandomForestClassifier#applying randomforestclassifier algorithm
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix#calculating the confusion matrix
cm = confusion_matrix(y_test, y_pred)


# Visualising the Train set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
a=np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01)
b=np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)


X1, X2 = np.meshgrid(a,b)# mesh grid will create a rentangular grid out of two given 1d arrays 
cll=classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)#ravel will convert ndim to 1dim array
plt.contourf(X1, X2,cll,alpha = 0.50, cmap = ListedColormap(('red', 'green')))
#contour graph is a way to represent a 3 dim surface on 2 dim surface plane
#it graphs two predicted variables X1 and X2 on y axis and a response variable Z as contours here Z is cll
#listedcolormap will fill color in the order of its list on map
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c = ListedColormap(('red', 'green'))(i), label = j)
    #scatter is used to plot data points on horizontal and vertical axis in the attempt to show how much one variable is affected by another
plt.title('Random Forest Classification (Training set)')
plt.xlabel('review_count')
plt.ylabel('Faker')
plt.legend()
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
cll1=classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
plt.contourf(X1, X2,cll1,
             alpha = 0.50, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('review_count')
plt.ylabel('Faker')
plt.legend()
plt.show()


