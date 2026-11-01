"""
coin_core.py
=============
Pure-Python engine for the Smart ATM Cash Dispenser (no GUI dependency, so it
can be unit-tested - see test_coin.py).

Problem (Minimum Coin / Note Change, Dynamic Programming):
    Given note denominations d1..dk and an amount A, pay exactly A using the
    FEWEST notes.

Recurrence:
    dp[0] = 0
    dp[a] = min over every note d <= a of ( dp[a - d] + 1 )     (INF if no note works)
    Remember which note gave the minimum (choice[a]) to trace the notes back.

Complexity (A = amount, k = number of denominations):
    Time  O(A * k)      Space O(A)
    Greedy (largest note first): O(k) time, but NOT always optimal.
    Limited stock (bounded variant, binary splitting): O(A * sum(log stock_i)) time.
"""

from __future__ import annotations

import re
from array import array
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

INF = float("inf")
MAX_AMOUNT = 1_000_000          # largest amount accepted by the unlimited DP
MAX_AMOUNT_LIMITED = 200_000    # largest amount accepted by the stock-limited DP


# --------------------------------------------------------------------------
# Input parsing / validation
# --------------------------------------------------------------------------
def parse_int_list(text: str) -> List[int]:
    """'500, 200 100;50' -> [500, 200, 100, 50]."""
    parts = [p for p in re.split(r"[,\s;]+", text.strip()) if p]
    if not parts:
        raise ValueError("Please enter at least one value.")
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            raise ValueError(f"'{p}' is not a whole number.") from None
    return out


def validate_denominations(denoms: Sequence[int]) -> List[int]:
    """Check the list and return it sorted largest-first."""
    if not denoms:
        raise ValueError("Enter at least one denomination.")
    if any(d <= 0 for d in denoms):
        raise ValueError("Denominations must be positive whole numbers.")
    if len(set(denoms)) != len(denoms):
        raise ValueError("Duplicate denominations are not allowed.")
    return sorted(denoms, reverse=True)


def validate_amount(amount: int, cap: int = MAX_AMOUNT) -> None:
    if isinstance(amount, bool) or not isinstance(amount, int):
        raise ValueError("Amount must be a whole number.")
    if amount < 0:
        raise ValueError("Amount cannot be negative.")
    if amount > cap:
        raise ValueError(f"Amount is too large (maximum {cap:,}).")


# --------------------------------------------------------------------------
# Dynamic programming (unlimited notes of each kind)
# --------------------------------------------------------------------------
@dataclass
class DPStep:
    """What happened while filling one cell dp[amount]."""
    amount: int
    candidates: List[Tuple[int, Optional[int]]]   # (note, dp[a-note]+1) ; None = unreachable
    value: Optional[int]                          # dp[amount]; None = cannot be made
    used: int                                     # note chosen (0 if none)


@dataclass
class DPResult:
    amount: int
    denoms: List[int]            # largest first
    dp: list                     # dp[a] = fewest notes (INF if impossible)
    choice: List[int]            # choice[a] = last note used for the optimum
    steps: List[DPStep]          # empty unless record_steps=True

    @property
    def min_count(self) -> Optional[int]:
        v = self.dp[self.amount]
        return None if v == INF else int(v)

    def path(self) -> List[int]:
        """Cells visited by the traceback: amount, amount-d1, ..., 0."""
        if self.min_count is None:
            return []
        cells, a = [self.amount], self.amount
        while a > 0:
            a -= self.choice[a]
            cells.append(a)
        return cells

    @property
    def notes(self) -> Optional[Dict[int, int]]:
        """{note: how many}, largest note first; None if impossible."""
        if self.min_count is None:
            return None
        counts: Dict[int, int] = {}
        a = self.amount
        while a > 0:
            d = self.choice[a]
            counts[d] = counts.get(d, 0) + 1
            a -= d
        return dict(sorted(counts.items(), reverse=True))


def solve_dp(amount: int, denoms: Sequence[int], record_steps: bool = False) -> DPResult:
    """Bottom-up DP. Ties prefer the larger note (denominations tried largest first)."""
    denoms = validate_denominations(list(denoms))
    validate_amount(amount)
    dp = [INF] * (amount + 1)
    dp[0] = 0
    choice = [0] * (amount + 1)
    steps: List[DPStep] = []
    for a in range(1, amount + 1):
        best, used = INF, 0
        cands: List[Tuple[int, Optional[int]]] = []
        for d in denoms:
            if d > a:
                continue
            cand = dp[a - d] + 1            # INF + 1 stays INF
            if record_steps:
                cands.append((d, None if cand == INF else int(cand)))
            if cand < best:
                best, used = cand, d
        dp[a], choice[a] = best, used
        if record_steps:
            steps.append(DPStep(a, cands, None if best == INF else int(best), used))
    return DPResult(amount, denoms, dp, choice, steps)


# --------------------------------------------------------------------------
# Greedy (for comparison)
# --------------------------------------------------------------------------
def greedy_change(amount: int, denoms: Sequence[int]) -> Tuple[Dict[int, int], int]:
    """Always take the biggest note that fits. Returns (counts, remainder_left)."""
    denoms = validate_denominations(list(denoms))
    validate_amount(amount)
    counts: Dict[int, int] = {}
    rest = amount
    for d in denoms:
        if rest >= d:
            counts[d] = rest // d
            rest -= counts[d] * d
    return counts, rest


# --------------------------------------------------------------------------
# Limited stock (ATM has only so many notes of each kind)
# --------------------------------------------------------------------------
def solve_limited(amount: int, denoms: Sequence[int],
                  stocks: Sequence[int]) -> Optional[Dict[int, int]]:
    """
    Bounded variant: note i can be used at most stocks[i] times.
    Each stock is split into chunks 1,2,4,...,remainder (binary splitting) so
    the problem becomes a 0/1 knapsack over chunks. Returns {note: count} or
    None if the amount cannot be paid from the available stock.
    """
    if len(denoms) != len(stocks):
        raise ValueError("Give one stock value for every denomination.")
    if any(s < 0 for s in stocks):
        raise ValueError("Stock cannot be negative.")
    validate_denominations(list(denoms))
    validate_amount(amount, MAX_AMOUNT_LIMITED)
    pairs = sorted(zip(denoms, stocks), reverse=True)

    items: List[Tuple[int, int]] = []           # (note, how many notes in this chunk)
    for d, s in pairs:
        s = min(s, amount // d)
        k = 1
        while s > 0:
            take = min(k, s)
            items.append((d, take))
            s -= take
            k *= 2

    dp = [INF] * (amount + 1)
    dp[0] = 0
    flags = []
    for d, size in items:
        w = d * size
        fl = bytearray(amount + 1)
        for a in range(amount, w - 1, -1):
            cand = dp[a - w] + size
            if cand < dp[a]:
                dp[a] = cand
                fl[a] = 1
        flags.append(fl)

    if dp[amount] == INF:
        return None
    counts: Dict[int, int] = {}
    a = amount
    for i in range(len(items) - 1, -1, -1):
        if flags[i][a]:
            d, size = items[i]
            counts[d] = counts.get(d, 0) + size
            a -= d * size
    return dict(sorted(counts.items(), reverse=True))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def format_notes(counts: Optional[Dict[int, int]], symbol: str = "") -> str:
    if counts is None:
        return "cannot be paid"
    if not counts:
        return "no notes needed"
    return " + ".join(f"{n} x {symbol}{d}" for d, n in counts.items())


def total_notes(counts: Dict[int, int]) -> int:
    return sum(counts.values())
