import time
import gurobipy as gp
from gurobipy import GRB
from typing import List, Optional, Dict, Tuple

# -----------------------------
# 비용(목적함수) 계산 (가중치/셋업 선택적)
# -----------------------------
def total_weighted_tardiness(order, p, d, w=None, s=None):
    if w is None:
        w = [1.0] * len(p)
    t, tot = 0, 0.0
    for idx, j in enumerate(order):
        if s is not None and idx > 0:
            t += s[order[idx-1]][j]   # setup
        t += p[j]                     # proc
        Tj = max(0, t - d[j])
        tot += w[j] * Tj
    return tot

# -----------------------------
# 이웃: swap, l-block insertion (논문 3.2)
# -----------------------------
def neighbors_swap(order):
    n = len(order)
    for i in range(n-1):
        for j in range(i+1, n):
            new_order = order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            yield ("swap", i, j, new_order)

def neighbors_lblock_forward(order, l):
    # (i, j): [i..i+l-1] 블록을 j 직후에 삽입 (i < j)
    n = len(order)
    for i in range(0, n - l):
        for j in range(i + l, n):
            blk = order[i:i+l]
            rest = order[:i] + order[i+l:]
            new_order = rest[:j - l + 1] + blk + rest[j - l + 1:]
            yield ("lF", i, j, l, new_order)

def neighbors_lblock_backward(order, l):
    # (i, j): [i..i+l-1] 블록을 j 위치 앞으로 이동 (j < i)
    n = len(order)
    for i in range(1, n - l + 1):
        for j in range(0, i):
            blk = order[i:i+l]
            rest = order[:i] + order[i+l:]
            new_order = rest[:j] + blk + rest[j:]
            yield ("lB", i, j, l, new_order)

# -----------------------------
# Δsetup(설정 변화) 계산식 — 논문 3.2의 식을 그대로 코딩
# (인덱스 경계는 None 처리로 0 setup로 둠)
# -----------------------------
def _S(s, a, b):
    if s is None: return 0
    if a is None or b is None: return 0
    return s[a][b]

def setup_variation_swap(order, i, j, s):
    # 논문 Alg.3 줄5의 식 (π 인덱스에 대응)
    n = len(order)
    ai = order[i]; aj = order[j]
    im1 = order[i-1] if i-1 >= 0 else None
    ip1 = order[i+1] if i+1 < n else None
    jm1 = order[j-1] if j-1 >= 0 else None
    jp1 = order[j+1] if j+1 < n else None

    old = _S(s, im1, ai) + _S(s, ai, ip1) + _S(s, jm1, aj) + _S(s, aj, jp1)
    new = _S(s, im1, aj) + _S(s, aj, ip1) + _S(s, jm1, ai) + _S(s, ai, jp1)
    return new - old

def setup_variation_lblock_forward(order, i, j, l, s):
    # 논문 Alg.6 줄5의 식
    n = len(order)
    a  = order[i]
    al = order[i+l-1]
    jm = order[j]       if j >= 0 and j < n else None
    jp = order[j+1]     if j+1 < n        else None
    im1 = order[i-1]    if i-1 >= 0       else None
    ip  = order[i+l]    if i+l < n        else None

    old = _S(s, im1, a) + _S(s, al, ip) + _S(s, jm, jp)
    new = _S(s, im1, ip) + _S(s, jm, a) + _S(s, al, jp)
    return new - old

def setup_variation_lblock_backward(order, i, j, l, s):
    # 논문 Alg.6 줄19의 식
    n = len(order)
    j0 = order[j]
    im1 = order[i-1]     if i-1 >= 0 else None
    a  = order[i]
    al = order[i+l-1]
    ip1 = order[i+1]     if i+1 < n else None
    jp1 = order[j+1]     if j+1 < n else None

    old = _S(s, order[j-1] if j-1>=0 else None, j0) + _S(s, im1, a) + _S(s, al, ip1)
    new = _S(s, order[j-1] if j-1>=0 else None, a) + _S(s, al, j0) + _S(s, im1, ip1)
    return new - old

# -----------------------------
# 초기해: EDD / SPT
# -----------------------------
def initial_order(p, d, rule="EDD"):
    if rule.upper() == "SPT":
        return sorted(range(len(p)), key=lambda j: p[j])
    return sorted(range(len(p)), key=lambda j: d[j])

# -----------------------------
# 학습 단계 (learning phase)
#   - 필터 없이 이웃을 훑으며 "개선 move의 Δsetup" 샘플 수집
#   - 각 이웃 유형별 리스트 정렬 뒤 θ 분위값을 maxΔs로 채택
# -----------------------------
def learn_thresholds(order, p, d, w, s,
                     L_values=(1,2,3),    # l-block 길이 후보
                     max_samples=2000,    # 샘플 상한 (속도/품질 트레이드오프)
                     theta=0.90):
    cur_val = total_weighted_tardiness(order, p, d, w, s)
    samples: Dict[str, List[int]] = {"swap": [], "lF": [], "lB": []}
    checked = 0

    # swap
    for tag, i, j, new_ord in neighbors_swap(order):
        dv = setup_variation_swap(order, i, j, s)
        new_val = total_weighted_tardiness(new_ord, p, d, w, s)
        if new_val < cur_val:
            samples["swap"].append(dv)
        checked += 1
        if checked >= max_samples: break

    # l-block forward/back
    for l in L_values:
        for tag, i, j, L, new_ord in neighbors_lblock_forward(order, l):
            dv = setup_variation_lblock_forward(order, i, j, L, s)
            new_val = total_weighted_tardiness(new_ord, p, d, w, s)
            if new_val < cur_val:
                samples["lF"].append(dv)
            checked += 1
            if checked >= max_samples: break
        if checked >= max_samples: break

    if checked < max_samples:  # 여유가 있으면 backward도
        for l in L_values:
            for tag, i, j, L, new_ord in neighbors_lblock_backward(order, l):
                dv = setup_variation_lblock_backward(order, i, j, L, s)
                new_val = total_weighted_tardiness(new_ord, p, d, w, s)
                if new_val < cur_val:
                    samples["lB"].append(dv)
                checked += 1
                if checked >= max_samples: break
            if checked >= max_samples: break

    # θ 분위수 → maxΔs
    def quantile(arr, th):
        if not arr:           # 학습샘플이 비어있으면 필터 끔(=무한대)
            return float("inf")
        arr = sorted(arr)
        pos = int(th * len(arr))
        if pos >= len(arr): pos = len(arr)-1
        return arr[pos]

    thresholds = {
        "swap": quantile(samples["swap"], theta),
        "lF":   quantile(samples["lF"],   theta),
        "lB":   quantile(samples["lB"],   theta),
    }
    return thresholds

# -----------------------------
# MILP 모델 (S, C, T, x)
# -----------------------------
def build_model(p: List[int], d: List[int]):
    n = len(p)
    J = range(n)
    P = sum(p)
    m = gp.Model("1||sumTj_withFilteringHeur")
    S = m.addVars(J, lb=0.0, name="S")
    C = m.addVars(J, lb=0.0, name="C")
    T = m.addVars(J, lb=0.0, name="T")
    x = m.addVars(J, J, vtype=GRB.BINARY, name="x")
    m.setObjective(gp.quicksum(T[j] for j in J), GRB.MINIMIZE)

    for j in J:
        m.addConstr(C[j] == S[j] + p[j])
        m.addConstr(T[j] >= C[j] - d[j])
        for k in J:
            if j == k:
                m.addConstr(x[j,j] == 0)
            else:
                m.addConstr(S[j] >= S[k] + p[k] - P * x[j,k])
    for j in J:
        for k in range(j+1, n):
            m.addConstr(x[j,k] + x[k,j] == 1)
    return m, S, C, T, x

# -----------------------------
# (order) -> (S,C,T,x) 값 만들기 (cbSetSolution 주입용)
# -----------------------------
def order_to_values(order, p, d):
    n = len(order)
    pos = {order[t]: t for t in range(n)}
    Sval, Cval = [0.0]*n, [0.0]*n
    t = 0.0
    for idx, j in enumerate(order):
        Sval[j] = t
        t += p[j]
        Cval[j] = t
    Tval = [max(0.0, Cval[j] - d[j]) for j in range(n)]
    Xval = {}
    for j in range(n):
        for k in range(n):
            if j == k: Xval[(j,k)] = 0.0
            else:
                # j가 k 뒤라면 x[j,k]=1 → S[j] >= S[k]+p[k] 강제
                Xval[(j,k)] = 1.0 if pos[j] > pos[k] else 0.0
    return Sval, Cval, Tval, Xval

# -----------------------------
# Heuristic Callback: 제한 필터 적용한 이웃만 평가, 개선 시 주입
# -----------------------------
def make_callback(p, d, w=None, s=None, L_values=(1,2,3),
                  theta=0.90, batch_per_call=100):
    # 초기해 + 학습단계로 임계치 산정 (논문 3.1)
    cur = initial_order(p, d, "EDD")
    thresholds = learn_thresholds(cur, p, d, w, s, L_values=L_values, theta=theta)

    cur_val = total_weighted_tardiness(cur, p, d, w, s)
    best = cur[:]; best_val = cur_val
    last_t = time.time()

    def callback(model, where):
        nonlocal cur, cur_val, best, best_val, last_t
        # 너무 자주 실행하면 solver 방해 → 간헐 실행
        if where != GRB.Callback.MIPNODE:
            return
        now = time.time()
        if now - last_t < 0.1:
            return
        last_t = now

        tried = 0
        # 1) swap
        for tag, i, j, new_ord in neighbors_swap(cur):
            if tried >= batch_per_call: break
            dv = setup_variation_swap(cur, i, j, s)
            if dv > thresholds["swap"]:  # 필터: setup 증가가 임계 초과면 skip
                continue
            new_val = total_weighted_tardiness(new_ord, p, d, w, s)
            if new_val < cur_val:
                _inject(model, new_ord, p, d)
                cur, cur_val = new_ord, new_val
                if cur_val < best_val:
                    best, best_val = cur[:], cur_val
                break
            tried += 1

        if tried < batch_per_call:
            # 2) l-block forward/back
            for l in L_values:
                # forward
                for tag, i, j, L, new_ord in neighbors_lblock_forward(cur, l):
                    if tried >= batch_per_call: break
                    dv = setup_variation_lblock_forward(cur, i, j, L, s)
                    if dv > thresholds["lF"]:
                        continue
                    new_val = total_weighted_tardiness(new_ord, p, d, w, s)
                    if new_val < cur_val:
                        _inject(model, new_ord, p, d)
                        cur, cur_val = new_ord, new_val
                        if cur_val < best_val:
                            best, best_val = cur[:], cur_val
                        return
                    tried += 1
                # backward
                for tag, i, j, L, new_ord in neighbors_lblock_backward(cur, l):
                    if tried >= batch_per_call: break
                    dv = setup_variation_lblock_backward(cur, i, j, L, s)
                    if dv > thresholds["lB"]:
                        continue
                    new_val = total_weighted_tardiness(new_ord, p, d, w, s)
                    if new_val < cur_val:
                        _inject(model, new_ord, p, d)
                        cur, cur_val = new_ord, new_val
                        if cur_val < best_val:
                            best, best_val = cur[:], cur_val
                        return
                    tried += 1
    return callback

def _inject(model, order, p, d):
    Sval, Cval, Tval, Xval = order_to_values(order, p, d)
    S = {v.VarName.split('[')[1].split(']')[0]: v for v in model.getVars() if v.VarName.startswith('S[')}
    C = {v.VarName.split('[')[1].split(']')[0]: v for v in model.getVars() if v.VarName.startswith('C[')}
    T = {v.VarName.split('[')[1].split(']')[0]: v for v in model.getVars() if v.VarName.startswith('T[')}
    X = {v.VarName.split('[')[1].split(']')[0]: v for v in model.getVars() if v.VarName.startswith('x[')}
    var_list, val_list = [], []
    n = len(p)
    for j in range(n):
        var_list += [S[str(j)], C[str(j)], T[str(j)]]
        val_list += [Sval[j],   Cval[j],   Tval[j]]
    for j in range(n):
        for k in range(n):
            var_list.append(X[f"{j},{k}"])
            val_list.append(Xval[(j,k)])
    model.cbSetSolution(var_list, val_list)

# -----------------------------
# 메인 실행 함수
# -----------------------------
def solve_with_filtering_limit_strategy(p: List[int],
                                       d: List[int],
                                       w: Optional[List[float]] = None,
                                       s: Optional[List[List[int]]] = None,
                                       theta: float = 0.90,
                                       L_values: Tuple[int, ...] = (1,2,3),
                                       time_limit: Optional[float] = None):
    m, S, C, T, X = build_model(p, d)
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    m.Params.OutputFlag = 1

    cb = make_callback(p, d, w=w, s=s, L_values=L_values, theta=theta, batch_per_call=100)
    m.optimize(cb)

    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        J = range(len(p))
        schedule = sorted(J, key=lambda j: S[j].X)
        print("\n=== 실행 결과 ===")
        print("작업 순서:", " -> ".join(str(j+1) for j in schedule))
        print("Total tardiness:", m.ObjVal)
        return schedule, m.ObjVal
    else:
        print("No feasible solution")
        return None, None


# ====== 예시 ======
if __name__ == "__main__":
    # Unweighted, setup 없음: 논문 전략을 그대로 적용하되 s=None로 동작
    p = [2, 4, 3]
    d = [3, 2, 6]
    solve_with_filtering_limit_strategy(p, d, w=None, s=None, theta=0.90, L_values=(1,2,3), time_limit=5.0)
