import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

#splitting data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

#create and train the model
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

#testing or evaluate the model
acc = linear.score(x_test, y_test) #models accuracy score
print(acc)
print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions= linear.predict(x_test)

for x in range(len(predictions)):
    print("Mark prediction: ",predictions[x],"\n", x_test[x], "\n","Actual mark: ",y_test[x])
    