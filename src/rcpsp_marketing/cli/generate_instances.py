import random
import re
import math
from typing import Dict, List, Tuple

# Диапазоны цен на ресурсы (за 1 ед. ресурса в 1 ед. времени)
# Ключи — индексы ресурсов из PSPLib (R1..Rk → 1..k)
RESOURCE_COST_RANGES = {
    1: (100.0, 250.0),   # R1: труд/дни разработки
    2: (80.0, 200.0),    # R2: маркетинг/баинг
    3: (50.0, 120.0),    # R3: инфраструктура/серверы
    4: (30.0, 80.0),     # R4: прочие/накладные
    # при наличии >4 ресурсов диапазоны для 5+ сгенерируются автоматически
}

# Вероятности типов задач
JOB_TYPES_PROBS = {
    "marketing": 0.40,
    "product":   0.25,
    "opt":       0.20,
    "partner":   0.10,
    "internal":  0.05,
}

# Эффекты типов (мультипликаторы вокруг 1.0 с шумом)
# Запись: var -> (mean_mult, std_of_noise), где noise ~ N(0, std)
EFFECTS = {
    "marketing": {"AC": (1.08, 0.05), "CPC": (1.04, 0.03), "LCR": (0.99, 0.01), "PCR": (0.99, 0.01), "LT": (1.00, 0.01)},
    "product":   {"AC": (1.01, 0.02), "CPC": (1.00, 0.01), "LCR": (1.04, 0.03), "PCR": (1.05, 0.03), "LT": (1.01, 0.01)},
    "opt":       {"AC": (1.00, 0.01), "CPC": (0.94, 0.03), "LCR": (1.01, 0.01), "PCR": (1.00, 0.01), "LT": (1.00, 0.01)},
    "partner":   {"AC": (1.03, 0.02), "CPC": (1.00, 0.01), "LCR": (1.01, 0.02), "PCR": (1.01, 0.02), "LT": (1.00, 0.01)},
    "internal":  {"AC": (1.00, 0.005), "CPC": (1.00, 0.005), "LCR": (1.00, 0.005), "PCR": (1.00, 0.005), "LT": (1.00, 0.005)},
}

# Коридоры значений (клипперы), чтобы метрики не «улетали»
BOUNDS = {
    "LT":  (50.0, 1500.0),
    "AC":  (0,  10_000_000),
    "CPC": (0.05, 5.0),
    "LCR": (0.005, 0.50),
    "PCR": (0.01, 0.60),
}

# Базовые «реалистичные» медианы для инициализации
INIT_MEDIANS = {
    "LT": 220.0,
    "AC": 1200.0,
    "CPC": 0.55,
    "LCR": 0.12,
    "PCR": 0.18,
}

# Степень масштабирования эффекта длительностью: scale = (dur/median_dur)^DUR_POW
DUR_POW = 0.5

# Концентрации для бета-распределений (чем больше, тем уже разброс)
BETA_KAPPA = 90.0

# Интенсивность насыщения рынка (чем больше, тем сильнее «тормозит» рост AC)
# Реальный K рынка возьмем пропорционально стартовому AC (ниже)
MARKET_K_MULTIPLIER = 250.0


# ============================================================
# Вспомогательные функции распределений и клипов
# ============================================================

def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sample_lognormal(median: float, iqr_ratio: float = 1.8) -> float:
    # iqr_ratio ~ e^{1.349*σ} → σ ≈ ln(iqr_ratio)/1.349
    sigma = math.log(iqr_ratio)/1.349
    mu = math.log(median)
    return math.exp(random.gauss(mu, sigma))

def sample_beta(mean: float, kappa: float = BETA_KAPPA) -> float:
    mean = clip(mean, 1e-6, 1 - 1e-6)
    alpha = mean * kappa
    beta = (1 - mean) * kappa
    x = random.gammavariate(alpha, 1.0)
    y = random.gammavariate(beta, 1.0)
    return x/(x+y)


# ============================================================
# Генерация начальных метрик (реалистичные распределения)
# ============================================================

def generate_initial_metrics() -> Dict[str, float]:
    LT  = clip(sample_lognormal(INIT_MEDIANS["LT"], 1.9), *BOUNDS["LT"])
    AC  = int(clip(sample_lognormal(INIT_MEDIANS["AC"], 2.2), *BOUNDS["AC"]))
    CPC = clip(sample_lognormal(INIT_MEDIANS["CPC"], 1.8), *BOUNDS["CPC"])
    LCR = clip(sample_beta(INIT_MEDIANS["LCR"], BETA_KAPPA), *BOUNDS["LCR"])
    PCR = clip(sample_beta(INIT_MEDIANS["PCR"], BETA_KAPPA), *BOUNDS["PCR"])
    CAC = CPC / max(1e-9, (LCR * PCR))
    margin = LT - CAC
    return {
        "LT": round(LT, 2),
        "AC": AC,
        "CPC": round(CPC, 3),
        "LCR": round(LCR, 4),
        "PCR": round(PCR, 4),
        "CAC": round(CAC, 2),
        "MARGIN": round(margin, 2),
    }


# ============================================================
# Парсинг PSPLib (минимально-надёжный)
# ============================================================

def parse_jobs_count(content: str) -> int:
    m = re.search(r'jobs.*?:\s*(\d+)', content, flags=re.I)
    if m:
        return int(m.group(1))
    m2 = re.search(r'number\s+of\s+activities.*?:\s*(\d+)', content, flags=re.I)
    if m2:
        return int(m2.group(1))
    # дефолт
    return 32

def parse_num_renewable_resources(content: str) -> int:
    m = re.search(r'renewable\s*resources.*?:\s*(\d+)', content, flags=re.I)
    if m:
        return int(m.group(1))
    m2 = re.search(r'-\s*renewable\s*:?\s*(\d+)', content, flags=re.I)
    if m2:
        return int(m2.group(1))
    return 4


def extract_requests_durations_block(content: str) -> List[str]:
    """
    Возвращает строки табличного блока REQUESTS/DURATIONS (если найден).
    """
    start = re.search(r'REQUESTS/DURATIONS', content, flags=re.I)
    if not start:
        return []
    # берём до следующего заголовка или конца файла
    tail = content[start.end():]
    stop = re.search(r'(PRECEDENCE|RESOURCEAVAILABILITIES|PROJECT|NR\s*=\s*|^\*{3,})', tail, flags=re.I | re.M)
    block = tail[:stop.start()] if stop else tail
    # отфильтруем строки с числами
    lines = [ln.strip() for ln in block.splitlines() if re.search(r'\d', ln)]
    return lines

def parse_requests_durations(content: str):
    durations = {}
    reqs = {}
    R = parse_num_renewable_resources(content)

    lines = extract_requests_durations_block(content)
    if not lines:
        return durations, reqs

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue

        # пропускаем заголовки/линии-разделители
        low = ln.lower()
        if "jobnr" in low or "mode" in low or "duration" in low:
            continue
        if set(ln) <= set("- "):
            continue

        ints = re.findall(r'-?\d+', ln)
        if not ints:
            continue

        # Формат PSPLIB обычно: job, mode, duration, R1..Rk
        if len(ints) >= 3 + R:
            j = int(ints[0])
            mode = int(ints[1])  # можно не использовать, но полезно для контроля
            d = int(ints[2])
            res = list(map(int, ints[3:3 + R]))

        # Иногда встречается укороченный формат: job, duration, R1..Rk
        elif len(ints) >= 2 + R:
            j = int(ints[0])
            d = int(ints[1])
            res = list(map(int, ints[2:2 + R]))
        else:
            continue

        durations[j] = d
        reqs[j] = {i + 1: res[i] for i in range(R)}

    return durations, reqs

# ============================================================
# Стоимости ресурсов и задач
# ============================================================

def generate_resource_costs(num_resources: int) -> Dict[int, float]:
    costs = {}
    for i in range(1, num_resources + 1):
        if i in RESOURCE_COST_RANGES:
            lo, hi = RESOURCE_COST_RANGES[i]
        else:
            # для «неожиданно» больших R — разумный дефолт
            lo, hi = 40.0, 120.0
        costs[i] = round(random.uniform(lo, hi), 2)
    return costs

def compute_task_cost(duration: int, usage: Dict[int, int], resource_costs: Dict[int, float]) -> float:
    total = 0.0
    if duration is None or duration <= 0:
        return 0.0
    for r, amount in (usage or {}).items():
        total += float(amount) * resource_costs.get(r, 0.0) * float(duration)
    return round(total, 2)


# ============================================================
# Эффекты задач (запись в файл в %), масштабирование, насыщение
# ============================================================

def choose_job_type() -> str:
    r = random.random()
    s = 0.0
    for jt, p in JOB_TYPES_PROBS.items():
        s += p
        if r <= s:
            return jt
    return "internal"

def duration_scale(job: int, durations: Dict[int, int]) -> float:
    if job not in durations or not durations:
        return 1.0
    d = durations[job]
    med = sorted(durations.values())[len(durations)//2]
    if med <= 0:
        return 1.0
    return (d / med) ** DUR_POW

def draw_effect_pct(var: str, jt: str, dur_scale: float) -> float:
    mean_mult, std = EFFECTS[jt][var]
    noise = random.gauss(0.0, std)  # шум «вокруг» (mean_mult-1.0)
    factor = (mean_mult - 1.0) + noise
    return round(100.0 * factor * dur_scale, 2)

def diminishing_AC_factor(ac_now: int, K: float) -> float:
    # чем больше AC относительно K, тем меньше чистый прирост
    return 1.0 / (1.0 + float(ac_now) / max(1.0, K))


# ============================================================
# Главная функция генерации
# ============================================================

def generate_psplib_with_metrics(input_file: str, output_file: str) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    n_jobs = parse_jobs_count(content)
    durations, reqs = parse_requests_durations(content)
    R = parse_num_renewable_resources(content)

    # 1) Инициал метрики
    M0 = generate_initial_metrics()
    K_market = max(1_000.0, float(M0["AC"]) * MARKET_K_MULTIPLIER)

    service_block = (
        "\nSERVICE METRICS (INITIAL):\n"
        f"LT_0    = {M0['LT']}\n"
        f"AC_0    = {M0['AC']}\n"
        f"CPC_0   = {M0['CPC']}\n"
        f"LCR_0   = {M0['LCR']}\n"
        f"PCR_0   = {M0['PCR']}\n"
        f"CAC_0   = {M0['CAC']}\n"
        f"MARGIN0 = {M0['MARGIN']}\n"
    )

    # 2) Стоимости ресурсов
    resource_costs = generate_resource_costs(R)
    res_costs_block = "\nRESOURCE COSTS (per unit per time):\n"
    for i in range(1, R+1):
        res_costs_block += f"R{i} = {resource_costs[i]:.2f}\n"

    # 3) Стоимость задач
    task_costs_block = "\nTASK COSTS (total):\n# jobnr : total_cost\n"
    # supersource (1) и supersink (n_jobs) — по нулям
    task_costs: Dict[int, float] = {}
    for j in range(1, n_jobs + 1):
        dur = durations.get(j, 0)
        usage = reqs.get(j, {})
        cost = compute_task_cost(dur, usage, resource_costs) if (j not in (1, n_jobs)) else 0.0
        task_costs[j] = cost
        task_costs_block += f"{j} : {cost:.2f}\n"

    # 4) Эффекты задач (в процентах, для чтения человеком/парсером)
    metric_changes_block = "\nMETRIC CHANGES (per job, multiplicative % effects):\n"
    metric_changes_block += "# jobnr : type        AC%    LT%    CPC%    LCR%    PCR%\n"
    metric_changes_block += "1 : supersource   0 0 0 0 0\n"

    # эффект генерируем для 2..n_jobs-1
    for j in range(2, n_jobs):
        jt = choose_job_type()
        dsc = duration_scale(j, durations)

        # рисуем «сырые» проценты (с учётом длительности),
        # насыщение AC фактически будет применяться уже в моделировании,
        # а здесь мы просто логируем базовый %-эффект
        acp  = draw_effect_pct("AC",  jt, dsc)
        ltp  = draw_effect_pct("LT",  jt, dsc)
        cpcp = draw_effect_pct("CPC", jt, dsc)
        lcrp = draw_effect_pct("LCR", jt, dsc)
        pcrp = draw_effect_pct("PCR", jt, dsc)

        metric_changes_block += f"{j} : {jt:<11} {acp:+.2f} {ltp:+.2f} {cpcp:+.2f} {lcrp:+.2f} {pcrp:+.2f}\n"

    metric_changes_block += f"{n_jobs} : supersink     0 0 0 0 0\n"

    # Финальная сборка
    new_content = (
        content.rstrip()
        + "\n"
        + service_block
        + res_costs_block
        + task_costs_block
        + metric_changes_block
        + "\nEND OF FILE\n"
        + ("*" * 72) + "\n"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"[ok] Файл с метриками и стоимостями создан: {output_file}")


# ============================================================
# Пример запуска
# ============================================================
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import random

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/extended")
    ap.add_argument("--pattern", default="*.sm")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    inp = Path(args.in_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # берём только файлы
    files = [p for p in sorted(inp.rglob(args.pattern)) if p.is_file()]
    print(f"[info] found {len(files)} files in {inp}")

    ok = 0
    fail = 0

    for p in files:
        # сохраняем структуру подпапок: raw/j30.sm/j301_1.sm -> extended/j30.sm/j301_1_with_metrics.sm
        rel = p.relative_to(inp)
        out_file = (out / rel).with_suffix("")  # убираем .sm
        out_file = out_file.with_name(out_file.name + "_with_metrics.sm")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            generate_psplib_with_metrics(str(p), str(out_file))
            ok += 1
        except Exception as e:
            print(f"[skip] {p}: {type(e).__name__}: {e}")
            fail += 1

    print(f"[done] ok={ok} fail={fail} out={out}")

