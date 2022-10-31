import pandas
import matplotlib.pyplot as py
from sklearn.linear_model import LinearRegression
data = pandas.read_csv('cost.csv')
d = data.describe()
print(d)
x = pandas.DataFrame(data, columns=['budget'])
y = pandas.DataFrame(data, columns=['world'])
py.figure(figsize=(10, 6,))
py.scatter(x, y, alpha=0.3)
py.xlabel('budget')
py.ylabel('world wide collection')
py.xlim(150000000, 500000000)
py.ylim(0, 3500000000)
re = LinearRegression()
re.fit(x, y)
py.plot(x, re.predict(x), color='red', linewidth=2)
py.show()
print(re.score(x, y))
