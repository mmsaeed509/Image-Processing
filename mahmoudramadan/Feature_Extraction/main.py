
import pandas as pd
dataset = pd.read_csv("Iris.csv")
print(dataset)
print("The number of rows is : ", dataset.shape[0])
print("The number of columns is : ", dataset.shape[1])
for col in dataset.columns:
    print(col)
print(dataset.head(5))
print(dataset.tail(10))
print(dataset.iloc[5:10, :1])
print(dataset[dataset.columns[0]][5:10])
print("The count of column SepalWidthCm : ", dataset['SepalWidthCm'].count())
dataset.rename(columns={'SepalLengthCm': 'Length'}, inplace=True)
print(dataset)



