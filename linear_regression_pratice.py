
import numpy as np
import pandas as pd


import matplotlib



housing = pd.read_csv('train.csv',names= [i for i in range(13)])
y = pd.read_csv('train2.csv',names= [0])
test = pd.read_csv('test.csv',names = [i for i in range(13)])





print(y.info())




from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()
min_max.fit(housing)
scalar_train = min_max.transform(housing)
scalar_train = pd.DataFrame(scalar_train,columns = housing.columns)
print(scalar_train)



mm2 =  MinMaxScaler()
mm2.fit(test)
scalar_test = mm2.transform(test)
scalar_test = pd.DataFrame(scalar_test,columns = test.columns)
print(scalar_test)





from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(scalar_train,y)
print(regr.coef_)



from sklearn.metrics import mean_squared_error
preds =  regr.predict(scalar_train)
print(preds)




mse =  mean_squared_error(preds,y)
print(mse)


import matplotlib.pyplot as plot
num = 100
x = np.arange(1,num+1)
plot.figure(figsize=(10,7))
plot.plot(x,y[:num],label='target')
plot.plot(x,preds[:num],label='predict')
plot.legend(loc = 'upper left')
plot.show()



result = regr.predict(scalar_test)
df_result = pd.DataFrame(result)
df_result
df_result.to_csv('result.csv',index=False,header=False)


regr.score(scalar_train,y)

