import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as  pt

df = pd.read_csv("dataset_ML.csv", sep=";")
# print(df)

x = np.array(df["X"]).reshape((-1, 1))
y = np.array(df["Y4"])

print(f"y = {y}", f"x = ", *x, sep=" ")

model_reg = LinearRegression().fit(x, y)
print(f"a_2 = {model_reg.coef_[0]} a_1 = {model_reg.intercept_}")

# predict
y_pred = model_reg.predict(np.array([2021, 2022]).reshape(-1, 1))
print("Прогноз: ", *y_pred, sep=" ")

print(pd.DataFrame(np.array([2021, 2022]), y_pred))

pt.plot(df["X"], df["Y4"], marker='o')

pt.plot(np.append(x, [[2021], [2022]]).reshape(-1, 1), model_reg.predict(np.append(x, [[2021], [2022]]).reshape(-1, 1)), marker='x')
pt.show()
