import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载历史数据
data = pd.read_csv('data.csv')

# 特征和标签
X = data.drop('配送时间', axis=1)  # 特征
y = data['配送时间']  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'reg:squarederror',  # 回归任务
    'max_depth': 3,                   # 树的最大深度
    'eta': 0.1,                       # 学习率
    'subsample': 0.8,                 # 子样本比例
    'colsample_bytree': 0.8,          # 特征采样比例
    'seed': 42                        # 随机种子
}

# 训练模型
num_round = 100  # 迭代次数
model = xgb.train(params, dtrain, num_round)

# 评估模型
y_pred = model.predict(dtest)
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

model.save_model('xgb_model.model')
print("模型保存到了：xgb_model.model")
