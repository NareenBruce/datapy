import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(10)  # Random x values
y = np.random.rand(10)  # Random y values
colors = np.random.rand(10)  # Random colors

plt.scatter(x, y, c=colors, cmap='viridis')  # Create a colored scatter plot
plt.colorbar()  # Add a color legend
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Colored Scatter Plot")
plt.show()

names= np.array(["Nareen","thiran"])
print("first name: ", names[0])
print("last name: ",names[-1])

numbers= np.array([[1,2,3],[4,5,6],[7,8,9]])
print("Second row third element: ", numbers[1][2])

newnumber=np.arange(1,26).reshape(5,5)
print("numbers are: \n", newnumber,"\n")

renumber=np.arange(1,11).reshape(2,-1)
print(renumber)

sample_dict={"x":1,"y":2,"z":3}
print(sample_dict)
the_series=pd.Series(sample_dict)
print(the_series)