from pulp import LpMaximize, LpProblem, LpVariable

model = LpProblem(name="生产计划", sense=LpMaximize)

# 决策变量
units_a = LpVariable(name="units_a", lowBound=0, cat="Continuous") # 产品A的单位数
units_b = LpVariable(name="units_b", lowBound=0, cat="Continuous") # 产品B的单位数

# 目标函数：最大化利润
model += 300 * units_a + 200 * units_b, "总利润"

# 约束条件
model += 2 * units_a + 1 * units_b <= 100, "处理时间"
model += 1 * units_a + 2 * units_b <=  80, "装配时间"

# 求解
model.solve()

print("生产计划：")
print(f"生产产品 A 单位数：{units_a.value()}")
print(f"生产产品 B 单位数：{units_b.value()}")
print(f"最大总利润：{model.objective.value()} 元")
