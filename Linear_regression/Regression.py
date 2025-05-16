import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("Linear_regression/student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])
print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

best= 0
for _ in range(30):

    #create and train the model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    # #testing or evaluate the model
    acc = linear.score(x_test, y_test) 

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
    
pickle_in = open("Linear_regression/studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

#Metrics
r2= linear.score(x_test, y_test)
predictions= linear.predict(x_test)
mse= mean_squared_error(y_test, predictions)
mae= mean_absolute_error(y_test, predictions)
rmse= np.sqrt(mse)
print("R2: ", r2)
print("MAE: ", mae)
print("RMSE: ", rmse)
print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)



for x in range(len(predictions)):
    print("Mark prediction: ",predictions[x],"\n", x_test[x], "\n","Actual mark: ",y_test[x])

plt.scatter(y_test, predictions, color='blue', edgecolors='k')
plt.xlabel("Actual Final Grade (G3)")
plt.ylabel("Predicted Final Grade")
plt.title("Predicted vs Actual Final Grades")
plt.grid(True)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Ideal line
plt.show()
