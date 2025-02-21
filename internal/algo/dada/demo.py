import xgboost as xgb
import pandas as pd

print("达达派单算法：有 3 个骑手和 1 个订单")

# 输入：订单特征、骑手特征/画像、环境特征等
# 输出：预测的配送时间 ETA
# 用途：为派单系统提供参考，帮助选择最优的骑手

# 特征数据
data = {
    '骑手到取货距离': [1.5, 0.8, 2.0],
    '骑手速度': [20, 15, 25],
    '骑手负载': [2, 1, 0],
    '骑手准时率': [95, 90, 98],
    '订单距离': [2.0, 1.5, 2.5],
    '订单金额': [50, 80, 30],
    '时间窗': [30, 45, 20],
    '天气': [0, 0, 1],
    '路况': [0, 1, 0],
    '时间段': [12, 12, 12],
    '骑手到取货时间': [4.5, 3.2, 4.8],
    '负载与金额交互': [100, 80, 0],
    '年龄': [25, 30, 28],
    '性别': [1, 0, 1],  # 男=1, 女=0
    '骑手等级': [2, 1, 3],  # 青铜=0, 白银=1, 黄金=2, 钻石=3
    '历史完成订单数': [500, 300, 800],
    '历史平均评分': [4.8, 4.5, 4.9],
    '历史投诉次数': [2, 5, 1],
    '是否全职': [1, 0, 1]  # 是=1, 否=0
}
df = pd.DataFrame(data)
print(df)

# 目标变量，即我们希望模型预测的值：从历史订单数据中提取，记录每个订单的实际配送时间
labels = [25, 30, 20]

# 转换为 DMatrix
dtrain = xgb.DMatrix(df, label=labels)

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

# 预测这一个订单如果分配给某个骑手预计配送时长
predictions = model.predict(dtrain)
print("历史结果:", labels)
print("预测结果:", predictions)

order_data = {
    '订单ID': ['A'],
    '取货地点经度': [116.40],
    '取货地点纬度': [39.90],
    '送货地点经度': [116.41],
    '送货地点纬度': [39.91],
    '订单金额': [50],
    '时间窗': [30]
}

rider_data = {
    '骑手ID': [1, 2, 3],
    '当前位置经度': [116.40, 116.41, 116.42],
    '当前位置纬度': [39.90, 39.91, 39.92],
    '当前速度': [20, 15, 25],
    '当前负载': [2, 1, 0],
    '历史准时率': [95, 90, 98]
}

# 转换为 DataFrame
order_df = pd.DataFrame(order_data)
rider_df = pd.DataFrame(rider_data)

# 计算骑手到取货地点的距离（假设使用欧氏距离）
def calculate_distance(lon1, lat1, lon2, lat2):
    return ((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2) ** 0.5

# 为每个骑手预测配送时间
predictions = []
for _, rider in rider_df.iterrows():
    # 构建特征数据
    features = {
        '骑手到取货距离': calculate_distance(rider['当前位置经度'], rider['当前位置纬度'],
                                           order_df['取货地点经度'][0], order_df['取货地点纬度'][0]),
        '骑手速度': rider['当前速度'],
        '骑手负载': rider['当前负载'],
        '骑手准时率': rider['历史准时率'],
        '订单距离': calculate_distance(order_df['取货地点经度'][0], order_df['取货地点纬度'][0],
                                     order_df['送货地点经度'][0], order_df['送货地点纬度'][0]),
        '订单金额': order_df['订单金额'][0],
        '时间窗': order_df['时间窗'][0],
        '天气': 0,  # 假设天气为晴
        '路况': 0,  # 假设路况畅通
        '时间段': 12,  # 假设为中午
        '骑手到取货时间': 4.5,  # 假设骑手到取货时间
        '负载与金额交互': 100,  # 假设负载与金额交互
        '年龄': 25,  # 假设年龄
        '性别': 1,  # 假设性别（男=1，女=0）
        '骑手等级': 2,  # 假设骑手等级（黄金=2）
        '历史完成订单数': 500,  # 假设历史完成订单数
        '历史平均评分': 4.8,  # 假设历史平均评分
        '历史投诉次数': 2,  # 假设历史投诉次数
        '是否全职': 1  # 假设是否全职（是=1，否=0）
    }
    # 转换为 DataFrame
    features_df = pd.DataFrame([features])
    # 确保列的顺序与训练时一致
    feature_names = [
        '骑手到取货距离', '骑手速度', '骑手负载', '骑手准时率', '订单距离', '订单金额', '时间窗', '天气', '路况', '时间段',
        '骑手到取货时间', '负载与金额交互', '年龄', '性别', '骑手等级', '历史完成订单数', '历史平均评分', '历史投诉次数', '是否全职'
    ]
    features_df = features_df[feature_names]
    # 预测配送时间
    prediction = model.predict(xgb.DMatrix(features_df))[0]
    predictions.append(prediction)

# 选择最优骑手
best_rider_index = predictions.index(min(predictions))
best_rider_id = rider_df.loc[best_rider_index, '骑手ID']
print(f"订单 A 分配给骑手 {best_rider_id}，预测配送时间: {predictions[best_rider_index]} 分钟")
