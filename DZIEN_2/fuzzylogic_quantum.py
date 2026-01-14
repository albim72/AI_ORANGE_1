"""
QuantumDice Training Planner (single-file)
------------------------------------------
Cel: poprawić jakość decyzji treningowych (i progresję) przez:
- wybór bodźca dnia na podstawie stanu organizmu i ograniczeń
- kontrolę ryzyka (kontuzja/zmęczenie)
- utrzymanie mikro-cyklu (objętość + 1 akcent + siła + long)
- "brutalną" dyscyplinę: plan ma rosnąć albo odpoczywasz mądrze

Wersja: v0.1 (czytelna, łatwa do modyfikacji)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import random
import datetime as dt


# ============================================================
# 1) MODELE DANYCH
# ============================================================

@dataclass
class AthleteState:
    """
    Stan bieżący sportowca.
    Wszystkie skale są umowne, ale spójne:
    - fatigue: 0..10 (0 świeży, 10 zmiażdżony)
    - soreness: 0..10 (DOMS/ból mięśni)
    - pain: 0..10 (ból urazowy, ostrzegawczy; 4+ traktuj jako alarm)
    - sleep_hours: realnie 0..10 (ile spałeś)
    - stress: 0..10
    - time_minutes: ile masz czasu dziś
    """
    fatigue: float
    soreness: float
    pain: float
    sleep_hours: float
    stress: float
    time_minutes: int

    # preferencje/cel
    goal: str = "ultra"  # "ultra" / "half" / "cut" etc.


@dataclass
class TrainingDayLog:
    date: dt.date
    session_id: str
    minutes: int
    intensity: str
    score: float
    risk: float
    rationale: str


@dataclass
class MicroCycleStats:
    """
    Śledzimy mikrocykl (7 dni): ile było biegania, siły, akcentów itd.
    To jest Twój "hamulec" na chaos.
    """
    run_days: int = 0
    strength_days: int = 0
    quality_days: int = 0   # interwały/tempo/podbiegi
    long_runs: int = 0
    total_minutes: int = 0

    last_quality_date: Optional[dt.date] = None
    last_long_date: Optional[dt.date] = None
    last_strength_date: Optional[dt.date] = None


@dataclass
class TrainingOption:
    """
    Kandydat na trening.
    - base_benefit: wartość adaptacyjna w idealnych warunkach
    - base_risk: ryzyko w idealnych warunkach
    - min_time: minimalny czas (min)
    - tags: do kontroli mikrocyklu
    """
    id: str
    name: str
    min_time: int
    minutes: int
    intensity: str  # "easy" / "moderate" / "hard" / "strength" / "rest"
    base_benefit: float
    base_risk: float
    tags: Tuple[str, ...] = field(default_factory=tuple)

    def is_quality(self) -> bool:
        return "quality" in self.tags

    def is_long(self) -> bool:
        return "long" in self.tags

    def is_strength(self) -> bool:
        return "strength" in self.tags


# ============================================================
# 2) QUANTUMDICE: OCENA + WYBÓR (REZONANS = SPRZĘŻENIE STANU I CELU)
# ============================================================

@dataclass
class QuantumDiceConfig:
    """
    - temperature: >0, im wyższa tym bardziej eksploruje
    - exploration: 0..1 domieszka eksploracji (epsilon-greedy style)
    - risk_aversion: jak mocno karzemy ryzyko
    """
    temperature: float = 0.65
    exploration: float = 0.12
    risk_aversion: float = 1.35
    seed: Optional[int] = None


class QuantumDice:
    """
    Minimalny silnik decyzyjny:
    1) liczy utility = benefit - risk_aversion * risk - penalties
    2) robi softmax (temperatura)
    3) z małym prawdopodobieństwem eksploruje
    4) zwraca najlepszą propozycję + TOP ranking i uzasadnienie
    """

    def __init__(self, config: QuantumDiceConfig):
        self.cfg = config
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)

    @staticmethod
    def _softmax(xs: List[float], temperature: float) -> List[float]:
        # stabilizacja numeryczna
        if temperature <= 1e-9:
            # temperatura ~0 => argmax deterministyczny
            m = max(xs)
            probs = [1.0 if x == m else 0.0 for x in xs]
            s = sum(probs)
            return [p / s for p in probs]

        m = max(xs)
        exps = [math.exp((x - m) / temperature) for x in xs]
        s = sum(exps)
        return [e / s for e in exps]

    def score_option(
        self,
        option: TrainingOption,
        state: AthleteState,
        stats: MicroCycleStats,
        today: dt.date
    ) -> Tuple[float, float, str]:
        """
        Zwraca: (utility_score, risk_score, rationale_short)
        """
        # ------------- constraint: czas
        if state.time_minutes < option.min_time:
            return -9999.0, 10.0, "ODRZUT: brak czasu"

        # ------------- ryzyko zależne od stanu
        # ból urazowy to czerwone światło
        pain_factor = 1.0 + (state.pain / 5.0)  # pain 5 => x2 ryzyko
        fatigue_factor = 1.0 + (state.fatigue / 10.0)  # fatigue 10 => x2
        sleep_penalty = max(0.0, 7.0 - state.sleep_hours) / 7.0  # poniżej 7h rośnie ryzyko

        risk = option.base_risk * pain_factor * fatigue_factor * (1.0 + sleep_penalty)

        # ------------- benefit zależny od celu
        benefit = option.base_benefit
        if state.goal == "ultra":
            # ultra: long + easy consistency
            if option.is_long():
                benefit *= 1.25
            if option.intensity == "easy":
                benefit *= 1.10
        elif state.goal == "half":
            # półmaraton: tempo i jakość
            if option.is_quality():
                benefit *= 1.25
            if option.intensity == "moderate":
                benefit *= 1.10
        elif state.goal == "cut":
            # redukcja: objętość nisko-średnia i siła
            if option.intensity in ("easy", "moderate"):
                benefit *= 1.10
            if option.is_strength():
                benefit *= 1.15

        # ------------- mikrocykl: brutalne zasady progresji i minimalnego sensu
        penalties = 0.0
        notes = []

        # 1) jeśli ból >= 4, zakaz "hard"/quality/long
        if state.pain >= 4.0:
            if option.is_quality() or option.is_long() or option.intensity == "hard":
                penalties += 4.0
                notes.append("kara: ból>=4 => zakaz jakości/long")

        # 2) minimum sensu tygodnia: 3 dni biegu + 1 siła (dla 50+ to ubezpieczenie)
        #    bonusujemy opcje, które domykają braki
        if stats.run_days < 3 and option.intensity in ("easy", "moderate", "hard"):
            benefit += 0.8
            notes.append("bonus: domyka 3 dni biegu/tydz")

        if stats.strength_days < 2 and option.is_strength():
            benefit += 0.9
            notes.append("bonus: domyka siłę")

        # 3) nie dawaj jakości dzień po dniu
        if option.is_quality() and stats.last_quality_date is not None:
            if (today - stats.last_quality_date).days < 2:
                penalties += 2.2
                notes.append("kara: jakość za szybko po jakości")

        # 4) long max 1x/tydz (tu w mikrocyklu)
        if option.is_long() and stats.long_runs >= 1:
            penalties += 2.5
            notes.append("kara: long już był w mikrocyklu")

        # 5) jeżeli zmęczenie wysokie, tniemy wartość ciężkich
        if state.fatigue >= 7.0 and option.intensity in ("hard", "moderate") and not option.is_strength():
            penalties += 1.8
            notes.append("kara: fatigue>=7 nie lubi mocnych akcentów")

        # 6) Jeżeli ostatnio biegałeś mało (np. run_days==0/1), karzemy hard
        if stats.run_days <= 1 and option.intensity == "hard":
            penalties += 2.0
            notes.append("kara: mała baza => hard ryzykowny")

        # ------------- ostateczny utility
        utility = benefit - self.cfg.risk_aversion * risk - penalties

        rationale = "; ".join(notes) if notes else "brak dodatkowych kar/bonusów"
        return utility, risk, rationale

    def choose(
        self,
        options: List[TrainingOption],
        state: AthleteState,
        stats: MicroCycleStats,
        today: dt.date,
        top_k: int = 6
    ) -> Tuple[TrainingOption, List[Tuple[float, TrainingOption, float, str]]]:
        """
        Zwraca:
        - wybraną opcję
        - ranking: (utility, option, risk, rationale)
        """
        scored = []
        for opt in options:
            u, r, why = self.score_option(opt, state, stats, today)
            scored.append((u, opt, r, why))

        scored.sort(key=lambda x: x[0], reverse=True)

        utilities = [u for (u, _, _, _) in scored]
        probs = self._softmax(utilities, self.cfg.temperature)

        # eksploracja: czasem losuj wg softmax, inaczej bierz top-1
        if random.random() < self.cfg.exploration:
            # wybór probabilistyczny
            pick = random.random()
            cum = 0.0
            chosen_idx = 0
            for i, p in enumerate(probs):
                cum += p
                if pick <= cum:
                    chosen_idx = i
                    break
            chosen = scored[chosen_idx][1]
        else:
            chosen = scored[0][1]

        return chosen, scored[:top_k]


# ============================================================
# 3) BIBLIOTEKA SESJI (opcje dnia)
# ============================================================

def default_training_options() -> List[TrainingOption]:
    """
    Zestaw bazowy, który możesz modyfikować.
    Min_time to próg wejścia, minutes to sugerowany czas.
    """
    return [
        TrainingOption(
            id="rest",
            name="Pełny odpoczynek + 20 min mobilizacji",
            min_time=20,
            minutes=20,
            intensity="rest",
            base_benefit=0.6,
            base_risk=0.05,
            tags=("recovery",)
        ),
        TrainingOption(
            id="easy_run_40",
            name="Bieg łatwy 40 min (Z2, luźno)",
            min_time=35,
            minutes=40,
            intensity="easy",
            base_benefit=2.6,
            base_risk=0.35,
            tags=("run",)
        ),
        TrainingOption(
            id="easy_run_60",
            name="Bieg łatwy 60 min (Z2, spokojnie)",
            min_time=55,
            minutes=60,
            intensity="easy",
            base_benefit=3.2,
            base_risk=0.45,
            tags=("run",)
        ),
        TrainingOption(
            id="moderate_hills",
            name="Podbiegi techniczne 10x45s + rozgrz/ schłodz (łącznie 55 min)",
            min_time=50,
            minutes=55,
            intensity="moderate",
            base_benefit=3.8,
            base_risk=0.85,
            tags=("run", "quality")
        ),
        TrainingOption(
            id="tempo_45",
            name="Tempo: 3x8 min (pomiędzy 3 min trucht), całość 50 min",
            min_time=45,
            minutes=50,
            intensity="hard",
            base_benefit=4.2,
            base_risk=1.05,
            tags=("run", "quality")
        ),
        TrainingOption(
            id="long_run_110",
            name="Długi bieg 110 min (Z2, kontrola, żywienie w trakcie)",
            min_time=95,
            minutes=110,
            intensity="easy",
            base_benefit=5.0,
            base_risk=1.10,
            tags=("run", "long")
        ),
        TrainingOption(
            id="strength_45",
            name="Siła 45 min (nogi + core + poślad + łydka, bez ego)",
            min_time=35,
            minutes=45,
            intensity="strength",
            base_benefit=3.0,
            base_risk=0.40,
            tags=("strength",)
        ),
        TrainingOption(
            id="easy_plus_str_70",
            name="Bieg łatwy 35 min + siła 30 min (łącznie ~65-70 min)",
            min_time=60,
            minutes=70,
            intensity="moderate",
            base_benefit=4.1,
            base_risk=0.75,
            tags=("run", "strength")
        ),
    ]


# ============================================================
# 4) AKTUALIZACJA STATYSTYK MIKROCYKLU
# ============================================================

def apply_session_to_stats(stats: MicroCycleStats, session: TrainingOption, date: dt.date) -> None:
    stats.total_minutes += session.minutes

    if "run" in session.tags or session.intensity in ("easy", "moderate", "hard"):
        # odróżniamy siłę od biegu
        if not session.is_strength() and session.intensity != "rest":
            stats.run_days += 1

    if session.is_strength():
        stats.strength_days += 1
        stats.last_strength_date = date

    if session.is_quality():
        stats.quality_days += 1
        stats.last_quality_date = date

    if session.is_long():
        stats.long_runs += 1
        stats.last_long_date = date


# ============================================================
# 5) RAPORT: "BRUTALNY" KOMENTARZ SYSTEMU
# ============================================================

def brutal_coach_comment(stats: MicroCycleStats) -> str:
    """
    To jest ta dosadna warstwa: bez poezji, tylko diagnoza.
    """
    bullets = []

    if stats.run_days < 3:
        bullets.append(f"- Bieganie {stats.run_days} dni/tydz: za mało. To nie buduje bazy, to ją podtrzymuje (albo i nie).")
    if stats.strength_days < 2:
        bullets.append(f"- Siła {stats.strength_days} dni/tydz: to jest poziom 'dla świętego spokoju', nie pod progres.")
    if stats.quality_days < 1:
        bullets.append("- Brak jakości: bez bodźca szybkości/siły biegowej stoisz w miejscu.")
    if stats.long_runs < 1:
        bullets.append("- Brak long run: pod ultra to jak budować most bez filaru środkowego.")

    if not bullets:
        return "W tym mikrocyklu robota wygląda sensownie. Teraz tylko nie zepsuj tego ego i chaosem."

    return "Diagnoza (bez cukru):\n" + "\n".join(bullets)


# ============================================================
# 6) SYMULACJA TYGODNIA: PRZYKŁAD
# ============================================================

def simulate_week(
    start_date: dt.date,
    states: List[AthleteState],
    cfg: QuantumDiceConfig
) -> List[TrainingDayLog]:
    """
    Podajesz listę stanów na 7 dni (lub ile chcesz).
    System wybiera sesję na każdy dzień.
    """
    qd = QuantumDice(cfg)
    options = default_training_options()
    stats = MicroCycleStats()
    logs: List[TrainingDayLog] = []

    for i, st in enumerate(states):
        day = start_date + dt.timedelta(days=i)
        chosen, top = qd.choose(options, st, stats, day)

        # score/risk jeszcze raz, żeby mieć dokładnie dla loga
        score, risk, why = qd.score_option(chosen, st, stats, day)

        rationale_lines = []
        rationale_lines.append(f"Wybrane: {chosen.name}")
        rationale_lines.append(f"Powód: {why}")
        rationale_lines.append("TOP kandydaci:")
        for rank, (u, opt, r, w) in enumerate(top, start=1):
            rationale_lines.append(f"  {rank}. u={u:.2f}, risk={r:.2f} | {opt.id} | {w}")

        logs.append(
            TrainingDayLog(
                date=day,
                session_id=chosen.id,
                minutes=chosen.minutes,
                intensity=chosen.intensity,
                score=float(score),
                risk=float(risk),
                rationale="\n".join(rationale_lines)
            )
        )

        apply_session_to_stats(stats, chosen, day)

    # Na koniec tygodnia: brutalny komentarz
    logs.append(
        TrainingDayLog(
            date=start_date + dt.timedelta(days=len(states)),
            session_id="WEEK_SUMMARY",
            minutes=stats.total_minutes,
            intensity="summary",
            score=0.0,
            risk=0.0,
            rationale=(
                f"Podsumowanie mikrocyklu:\n"
                f"- run_days={stats.run_days}\n"
                f"- strength_days={stats.strength_days}\n"
                f"- quality_days={stats.quality_days}\n"
                f"- long_runs={stats.long_runs}\n"
                f"- total_minutes={stats.total_minutes}\n\n"
                + brutal_coach_comment(stats)
            )
        )
    )

    return logs


# ============================================================
# 7) PRZYKŁADOWE URUCHOMIENIE
# ============================================================

if __name__ == "__main__":
    # Przykład: tydzień, w którym człowiek jest zmęczony, ma trochę stresu,
    # a czasem ma tylko 45-60 minut.
    start = dt.date.today()

    example_states = [
        AthleteState(fatigue=6.5, soreness=5.0, pain=2.0, sleep_hours=6.0, stress=6.0, time_minutes=50, goal="ultra"),
        AthleteState(fatigue=7.5, soreness=6.0, pain=3.0, sleep_hours=5.5, stress=7.0, time_minutes=40, goal="ultra"),
        AthleteState(fatigue=5.5, soreness=4.0, pain=2.0, sleep_hours=7.2, stress=5.0, time_minutes=70, goal="ultra"),
        AthleteState(fatigue=6.0, soreness=5.0, pain=4.5, sleep_hours=6.5, stress=6.0, time_minutes=60, goal="ultra"),  # ból>=4 => hamulec
        AthleteState(fatigue=4.5, soreness=3.5, pain=2.0, sleep_hours=7.8, stress=4.0, time_minutes=110, goal="ultra"),
        AthleteState(fatigue=6.0, soreness=5.0, pain=2.5, sleep_hours=6.8, stress=5.0, time_minutes=45, goal="ultra"),
        AthleteState(fatigue=5.0, soreness=4.0, pain=2.0, sleep_hours=7.0, stress=4.5, time_minutes=70, goal="ultra"),
    ]

    cfg = QuantumDiceConfig(
        temperature=0.60,    # mniej chaosu
        exploration=0.10,    # trochę eksploracji
        risk_aversion=1.45,  # 50+ => karz ryzyko mocniej
        seed=42
    )

    logs = simulate_week(start, example_states, cfg)

    for log in logs:
        print("\n" + "=" * 60)
        print(f"DATE: {log.date} | SESSION: {log.session_id} | {log.minutes} min | {log.intensity}")
        if log.session_id != "WEEK_SUMMARY":
            print(f"SCORE: {log.score:.2f} | RISK: {log.risk:.2f}")
        print(log.rationale)
