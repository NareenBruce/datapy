import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("Linear_regression/student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# best= 0
# for _ in range(30):
#     #splitting data into training and testing data
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

#     #create and train the model
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)

#     # #testing or evaluate the model
#     acc = linear.score(x_test, y_test) #models accuracy score
#     print(acc)

#     if acc > best:
#         best = acc
#         with open("studentmodel.pickle", "wb") as f:
#             pickle.dump(linear, f)
    
pickle_in = open("Linear_regression/studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)
acc = linear.score(x_test, y_test) #models accuracy score
print("Accuracy: ", acc)
print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions= linear.predict(x_test)

for x in range(len(predictions)):
    print("Mark prediction: ",predictions[x],"\n", x_test[x], "\n","Actual mark: ",y_test[x])

#setting and styling the graph
style.use("ggplot")
p="failures"
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
