import pandas as pd

train = pd.read_csv('orginal_train.csv',names=[i for i in range(14)])


train2 = train[1]
train2.to_csv('train2.csv',index = False,header= False)


train = train.drop(columns=1)

train.to_csv('train.csv',header = False, index = False)




