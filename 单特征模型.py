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

# 分割数据集
x = df_train["GrLivArea"].to_numpy().reshape(-1, 1)  # 从一维转为二维
y = df_train['SalePrice'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# 序列解包的形式得到：训练输入集，测试输入集，训练输出集，测试输出集
print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# 训练模型
lr = LinearRegression()  # 实例化线性回归模型
lr.fit(x_train, y_train)  # 拟合函数：训练模型（参数为训练集）

# 预测房价
y_pred = lr.predict(x_test)  # 预测函数：根据训练的结果对输入集进行结果预测

# 预测结果可视化
plt.scatter(x_train, y_train, color="green")  # 训练集的散点图
plt.scatter(x_test, y_test, color="blue")  # 测试集的散点图
plt.plot(x_test, y_pred, color="red")  # 测试输入集和预测结果的回归线
plt.show()

# 打印模型信息
print("系数：", lr.coef_)  # 返回一个列表数据，故实际计算时需通过索引获取，列表里表示模型系数，这里模型只有一个特征值，所以只有一个系数。
print("截距：", lr.intercept_)
print("模型：Scaleprice={}+{}*GrLivArea".format(lr.intercept_, lr.coef_[0]))
# 直线方程模型：Y = kX + b ，k为系数，b为截距
print("均方根误差：{}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
# 求解均方误差后再开方即得到均方根误差，使用均方根误差衡量模型好坏，数值越小越好。

# 批量预测实际的房价
areas = np.array([[1000], [2000], [3000]])
# 3组数据，每组数据只有一个数值，因为模型是单特征模型
prices = lr.predict(areas)
pred_prices = pd.DataFrame(np.c_[areas, prices], columns=[
                           "面积(平方英尺)", "价格(美元)"])

print(pred_prices)
