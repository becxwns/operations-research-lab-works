import random
import gurobipy as gp
from gurobipy import GRB

# === 데이터 생성 함수 ===
def generate_jobs(n, p_low=1, p_high=20, due_factor=1.2):
    """
    n: 작업 개수
    p_low, p_high: 처리시간 범위
    due_factor: 납기 설정 계수
    """
    p = [random.randint(p_low, p_high) for _ in range(n)]
    P_total = sum(p)
    d = [random.randint(int(P_total/n*0.5), int(P_total/n*due_factor)) for _ in range(n)]
    return p, d

# === 총 지연시간 최소화 모델 (Gurobi) ===
def min_total_tardiness(p, d):
    n = len(p)
    J = range(n)
    P = sum(p)

    m = gp.Model("SingleMachine_Tardiness")
    
    # 변수 정의
    S = m.addVars(J, lb=0.0, name="S")              # 시작시간
    C = m.addVars(J, lb=0.0, name="C")              # 완료시간
    T = m.addVars(J, lb=0.0, name="T")              # 지연시간
    x = m.addVars(J, J, vtype=GRB.BINARY, name="x") # 순서 결정 변수

    # 목적함수: 총 지연시간 최소화
    m.setObjective(gp.quicksum(T[j] for j in J), GRB.MINIMIZE)

    # 제약조건
    for j in J:
        m.addConstr(C[j] == S[j] + p[j])          # 완료시간 정의
        m.addConstr(T[j] >= C[j] - d[j])          # 지연시간 정의
        for k in J:
            if j == k:
                m.addConstr(x[j, j] == 0)
            else:
                m.addConstr(S[j] >= S[k] + p[k] - P * x[j, k])
    for j in J:
        for k in range(j+1, n):
            m.addConstr(x[j, k] + x[k, j] == 1)   # 순서 관계 강제

    # 최적화 실행
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        def start_time(job_index):
            return S[job_index].X

        schedule = sorted(J, key=start_time)
        obj = int(m.ObjVal)
        return schedule, obj
    else:
        return None, None

# === EDD 스케줄링 (Earliest Due Date) ===
def edd_schedule(p, d):
    n = len(p)
    jobs = list(range(n))

    def get_due_date(job_index):
        return d[job_index]

    jobs.sort(key=get_due_date)

    time = 0
    tardiness = 0
    for j in jobs:
        time += p[j]
        tardiness += max(0, time - d[j])

    return jobs, tardiness

# === SPT 스케줄링 (Shortest Processing Time) ===
def spt_schedule(p, d):
    n = len(p)
    jobs = list(range(n))

    def get_processing_time(job_index):
        return p[job_index]

    jobs.sort(key=get_processing_time)

    time = 0
    tardiness = 0
    for j in jobs:
        time += p[j]
        tardiness += max(0, time - d[j])

    return jobs, tardiness

# === 실행 예시: 대규모 문제 (100개 Job) ===
if __name__ == "__main__":
    # 1. 데이터 한 번만 생성
    p, d = generate_jobs(100)  # 100개 작업 생성
    print("처리시간 샘플:", p[:10], "...")
    print("납기일 샘플:", d[:10], "...")

    # 2. Gurobi 최적해
    opt_schedule, opt_obj = min_total_tardiness(p, d)
    print("\n[Gurobi 최적해]")
    if opt_schedule:
        print("총 지연시간:", opt_obj)
        print("작업 순서 (앞 20개):", " -> ".join(str(j+1) for j in opt_schedule[:20]), "...")

    # 3. EDD
    edd_seq, edd_obj = edd_schedule(p, d)
    print("\n[EDD 규칙]")
    print("총 지연시간:", edd_obj)
    print("작업 순서 (앞 20개):", " -> ".join(str(j+1) for j in edd_seq[:20]), "...")

    # 4. SPT
    spt_seq, spt_obj = spt_schedule(p, d)
    print("\n[SPT 규칙]")
    print("총 지연시간:", spt_obj)
    print("작업 순서 (앞 20개):", " -> ".join(str(j+1) for j in spt_seq[:20]), "...")

   