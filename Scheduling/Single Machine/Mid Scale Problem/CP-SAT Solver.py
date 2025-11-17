from ortools.sat.python import cp_model
import random

# === 데이터 생성 ===
def generate_jobs(n=12, p_low=1, p_high=20, due_factor=1.2):
    p = [random.randint(p_low, p_high) for _ in range(n)]
    P_total = sum(p)
    d = [random.randint(int(P_total/n*0.5), int(P_total/n*due_factor)) for _ in range(n)]
    return p, d

# === CP-SAT: Total Tardiness 최소화 ===
def min_total_tardiness_cpsat(p, d, time_limit_s=None):
    n = len(p)
    model = cp_model.CpModel()

    # 스케줄링 시간 상한(수평선)
    horizon = sum(p) + max(d)  # 여유 있게 설정

    # 변수: 시작/완료/지연/인터벌
    S = [model.NewIntVar(0, horizon, f"S[{j}]") for j in range(n)]
    C = [model.NewIntVar(0, horizon, f"C[{j}]") for j in range(n)]
    T = [model.NewIntVar(0, horizon, f"T[{j}]") for j in range(n)]
    intervals = [model.NewIntervalVar(S[j], p[j], C[j], f"I[{j}]") for j in range(n)]

    # 한 대의 기계: 작업들이 겹치지 않음
    model.AddNoOverlap(intervals)

    # 완료시간/지연시간 정의
    for j in range(n):
        # C[j] = S[j] + p[j] 는 IntervalVar가 이미 보장 (끝점이 end이며 길이가 processing)
        # 지연시간 T[j] ≥ C[j] - d[j]
        model.Add(T[j] >= C[j] - d[j])
        # T[j] ≥ 0  (변수 영역으로 이미 0..horizon)
        # model.Add(T[j] >= 0)  # 생략 가능

    # 목적함수: Σ T[j] 최소화
    model.Minimize(sum(T))

    # 풀이
    solver = cp_model.CpSolver()
    if time_limit_s is not None:
        solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 8  # 멀티코어 활용(원하면 조절)

    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # 시작시간 기준으로 순서 정렬(람다 없이)
        starts = [(solver.Value(S[j]), j) for j in range(n)]
        starts.sort()
        schedule = [j for _, j in starts]
        obj = int(solver.ObjectiveValue())

        print("\n=== 최적화 결과 (CP-SAT) ===")
        print("총 지연시간:", obj)
        print("작업 순서:", " -> ".join(str(j+1) for j in schedule))
        # 필요시 각 작업의 (시작, 완료, 납기, 지연) 출력
        # for j in schedule:
        #     print(f"Job {j+1}: S={solver.Value(S[j])}, C={solver.Value(C[j])}, d={d[j]}, T={solver.Value(T[j])}")
        return schedule, obj
    else:
        print("해를 찾지 못했습니다.")
        return None, None

# === 실행 예시 (중규모 12개) ===
if __name__ == "__main__":
    p, d = generate_jobs(12)
    print("처리시간:", p)
    print("납기일:", d)
    min_total_tardiness_cpsat(p, d)  # 필요하면 time_limit_s=10 같이 제한 가능
