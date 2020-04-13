import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.model_selection import train_test_split  # 分隔数据集
from sklearn.metrics import mean_squared_error  # 计算均方误差

# 加载数据
df_train = pd.read_csv('train.csv')

# 处理丢失数据
missing_data = df_train.isnull().sum().sort_values(ascending=False)
df_train = df_train.drop((missing_data[missing_data > 1]).index, axis=1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

# 处理离群值
df_train = df_train[df_train['GrLivArea'] < 4500]

# 重点关注字段
cols = ['OverallQual', 'GrLivArea', 'TotalBsmtSF',
        'GarageCars', 'FullBath', 'YearBuilt']
# 训练模型
X = df_train[['GrLivArea','TotalBsmtSF']].to_numpy()
y = df_train['SalePrice'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
fig = plt.figure(figsize = (8,8))
ax = fig.gca(projection = '3d')
ax.set_xlabel('OverallQual')
ax.set_ylabel('TotalBsmtSF')
ax.set_zlabel('SalePrice')
ax.scatter(X_test[:,0],X_test[:,1],y_test,color = 'red')
xx = X_test[:,0]
yy = X_test[:, 1]
XX,YY = np.meshgrid(np.linspace(xx.min(),xx.max()),np.linspace(yy.min(),yy.max()),)
ZZ = lr.predict(np.c_[XX.flatten(),YY.flatten()]).reshape(XX.shape)
ax.plot_surface(XX,YY,ZZ,alpha = 0.2)
plt.show()
# 打印模型信息
print("系数：", lr.coef_)  # 返回一个列表数据，故实际计算时需通过索引获取，列表里表示模型系数，这里模型只有一个特征值，所以只有一个系数。
print("截距：", lr.intercept_)
print("模型：Scaleprice={}+{}*GrLivArea + {} * TotalBsntSF".format(lr.intercept_,
                                                                lr.coef_[0], lr.coef_[1]))
# 直线方程模型：Y = kX + b ，k为系数，b为截距
print("均方根误差：{}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
# 求解均方误差后再开方即得到均方根误差，使用均方根误差衡量模型好坏，数值越小越好。
# 批量预测实际的房价
areas = np.array([[1500,1000], [2500,2000], [3500,3000]])
# 3组数据，每组数据只有一个数值，因为模型是单特征模型
prices = lr.predict(areas)
pred_prices = pd.DataFrame(np.c_[areas, prices], columns=[
                           "居住面积","地下室面积","房价"])

print(pred_prices)
