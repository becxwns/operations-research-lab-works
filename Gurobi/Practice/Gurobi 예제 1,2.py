import gurobipy as gp
from gurobipy import GRB

# 예제 1
# 모델 생성
model1 = gp.Model("mip1")

# 변수 정의
x = model1.addVar(vtype=GRB.BINARY, name="x")
y = model1.addVar(vtype=GRB.BINARY, name="y")
z = model1.addVar(vtype=GRB.BINARY, name="z")

# 제약조건 추가
model1.addConstr(x + 2 * y + 3 * z <= 4, name="c0")
model1.addConstr(x + y >= 1, name="c1")

# 목적 함수 설정 (최대화)
model1.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

# 최적화 실행
model1.optimize()

# 결과 출력
if model1.status == GRB.OPTIMAL:
    print(f"최적해 Objective Value: {model1.ObjVal:g}")
    for v in model1.getVars():  # getVars: 최적화 결과에서 변수 목록 가져오기
        print(f"{v.varName} = {v.x}")
    print("최적해를 찾았습니다.")
else:
    print("최적해를 찾지 못했거나 문제가 발생하였습니다.")

# 예제2
model2 = gp.Model("mip2")
x = model2.addVar(vtype=GRB.CONTINUOUS, name="x")
y = model2.addVar(vtype=GRB.CONTINUOUS, name="y")
z = model2.addVar(vtype=GRB.CONTINUOUS, name="z")

model2.addConstr(x - y <= 4, name = "c2")
model2.addConstr(x + y <= 4, name = "c3")
model2.addConstr(-0.25 * x + y <= 1, "c4")
model2.setObjective(y, GRB.MAXIMIZE)

model2.optimize()

if model2.status == GRB.OPTIMAL:
    print(f"최적해 Objective Value: {model2.ObjVal:g}")
    for v in model2.getVars():
        print(f"{v.varName} = {v.x}")
    print("최적해를 찾았습니다.")
else:
    print("최적해를 찾지 못했거나 문제가 발생했습니다")








