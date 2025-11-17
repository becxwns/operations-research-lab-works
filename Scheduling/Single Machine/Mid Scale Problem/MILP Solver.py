from ortools.linear_solver import pywraplp
import random

# === 데이터 생성 함수 ===
def generate_jobs(n=12, p_low=1, p_high=20, due_factor=1.2):
    """
    n: 작업 개수
    p_low, p_high: 처리시간 범위
    due_factor: 납기일 상한을 조절하는 계수
    """
    p = [random.randint(p_low, p_high) for _ in range(n)]
    P_total = sum(p)
    # 납기일: 평균 처리시간 기준 [0.5배 ~ due_factor배] 범위에서 랜덤
    d = [random.randint(int(P_total / n * 0.5), int(P_total / n * due_factor)) for _ in range(n)]
    return p, d

# === 총 지연시간 최소화 (MILP) ===
def min_total_tardiness_ortools(p, d):
    n = len(p)
    P = sum(p)

    # Solver 생성 (SCIP 사용)
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        print("Solver 로딩 실패")
        return None, None

    # === 변수 정의 ===
    S = [solver.NumVar(0, solver.infinity(), f"S[{j}]") for j in range(n)]   # 시작시간
    C = [solver.NumVar(0, solver.infinity(), f"C[{j}]") for j in range(n)]   # 완료시간
    T = [solver.NumVar(0, solver.infinity(), f"T[{j}]") for j in range(n)]   # 지연시간
    x = [[solver.IntVar(0, 1, f"x[{j},{k}]") for k in range(n)] for j in range(n)]  # 순서 변수

    # === 목적함수: 총 지연시간 최소화 ===
    solver.Minimize(solver.Sum(T[j] for j in range(n)))

    # === 제약조건 ===
    for j in range(n):
        solver.Add(C[j] == S[j] + p[j])          # 완료시간 정의
        solver.Add(T[j] >= C[j] - d[j])          # 지연시간 정의
        for k in range(n):
            if j == k:
                solver.Add(x[j][j] == 0)
            else:
                # 순서 제약 (Big-M 기법)
                solver.Add(S[j] >= S[k] + p[k] - P * x[j][k])

    # 두 작업 j, k 중 반드시 하나는 먼저 와야 함
    for j in range(n):
        for k in range(j + 1, n):
            solver.Add(x[j][k] + x[k][j] == 1)

    # === 최적화 실행 ===
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        # 결과 정리
        start_times = [(S[j].solution_value(), j) for j in range(n)]
        schedule = [job for _, job in sorted(start_times)]
        obj = solver.Objective().Value()

        print("\n=== 최적화 결과 (MILP, OR-Tools) ===")
        print("총 지연시간:", int(obj))
        print("작업 순서:", " -> ".join(str(j+1) for j in schedule))
        return schedule, obj
    else:
        print("해를 찾을 수 없습니다.")
        return None, None

# === 실행 예시 (중규모 문제 12개) ===
if __name__ == "__main__":
    p, d = generate_jobs(12)
    print("처리시간:", p)
    print("납기일:", d)
    min_total_tardiness_ortools(p, d)