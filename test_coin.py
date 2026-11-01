"""Unit tests for coin_core. Run:  python -m unittest -v test_coin"""

import itertools
import random
import time
import unittest
from collections import deque

import coin_core as cc

ATM = [500, 200, 100, 50, 10]


def bfs_min(amount, denoms):
    """Independent reference: breadth-first search over amounts."""
    if amount == 0:
        return 0
    seen = {0}
    q = deque([(0, 0)])
    while q:
        cur, n = q.popleft()
        for d in denoms:
            nxt = cur + d
            if nxt == amount:
                return n + 1
            if nxt < amount and nxt not in seen:
                seen.add(nxt)
                q.append((nxt, n + 1))
    return None


def brute_limited(amount, denoms, stocks):
    best = None
    for combo in itertools.product(*[range(s + 1) for s in stocks]):
        if sum(c * d for c, d in zip(combo, denoms)) == amount:
            n = sum(combo)
            best = n if best is None else min(best, n)
    return best


class TestBoundaryCases(unittest.TestCase):
    def test_amount_zero(self):
        r = cc.solve_dp(0, ATM)
        self.assertEqual(r.min_count, 0)
        self.assertEqual(r.notes, {})

    def test_amount_smaller_than_smallest_note(self):
        self.assertIsNone(cc.solve_dp(5, ATM).notes)

    def test_impossible_amount(self):
        r = cc.solve_dp(35, [10, 20])
        self.assertIsNone(r.min_count)
        self.assertIsNone(r.notes)
        self.assertEqual(r.path(), [])

    def test_single_denomination(self):
        self.assertEqual(cc.solve_dp(50, [10]).notes, {10: 5})
        self.assertIsNone(cc.solve_dp(55, [10]).notes)

    def test_amount_equals_one_note(self):
        self.assertEqual(cc.solve_dp(500, ATM).notes, {500: 1})

    def test_denomination_one_always_works(self):
        r = cc.solve_dp(37, [1, 5])
        self.assertEqual(r.min_count, 9)          # 7 x 5 + 2 x 1 = 37 -> 9 notes

    def test_large_amount(self):
        t0 = time.time()
        r = cc.solve_dp(100_000, ATM)
        self.assertEqual(r.notes, {500: 200})
        self.assertLess(time.time() - t0, 5)

    def test_maximum_amount_accepted(self):
        self.assertEqual(cc.solve_dp(cc.MAX_AMOUNT, ATM).min_count, 2000)

    def test_invalid_denominations(self):
        for bad in ([], [0, 10], [-5, 10], [10, 10, 20]):
            with self.assertRaises(ValueError):
                cc.solve_dp(100, bad)

    def test_invalid_amount(self):
        for bad in (-1, cc.MAX_AMOUNT + 1, 3.5, True):
            with self.assertRaises(ValueError):
                cc.solve_dp(bad, ATM)

    def test_parsing(self):
        self.assertEqual(cc.parse_int_list("500, 200 100;50"), [500, 200, 100, 50])
        for bad in ("", "  ", "10, abc", "1.5"):
            with self.assertRaises(ValueError):
                cc.parse_int_list(bad)


class TestCorrectness(unittest.TestCase):
    def test_typical_atm_withdrawal(self):
        r = cc.solve_dp(3770, ATM)
        self.assertEqual(r.min_count, 11)
        self.assertEqual(r.notes, {500: 7, 200: 1, 50: 1, 10: 2})

    def test_notes_add_up(self):
        for amount in (10, 70, 340, 1230, 9990):
            r = cc.solve_dp(amount, ATM)
            self.assertEqual(sum(d * n for d, n in r.notes.items()), amount)
            self.assertEqual(sum(r.notes.values()), r.min_count)

    def test_greedy_fails_case_1(self):
        counts, rest = cc.greedy_change(12, [1, 6, 10])
        self.assertEqual((rest, cc.total_notes(counts)), (0, 3))
        self.assertEqual(cc.solve_dp(12, [1, 6, 10]).min_count, 2)

    def test_greedy_fails_case_2(self):
        counts, rest = cc.greedy_change(30, [25, 10, 1])
        self.assertEqual(cc.total_notes(counts), 6)            # 25 + 5 x 1
        self.assertEqual(cc.solve_dp(30, [25, 10, 1]).min_count, 3)   # 3 x 10

    def test_greedy_can_fail_to_pay_at_all(self):
        counts, rest = cc.greedy_change(8, [5, 4])
        self.assertNotEqual(rest, 0)                            # 5 then stuck with 3
        self.assertEqual(cc.solve_dp(8, [5, 4]).min_count, 2)   # 4 + 4

    def test_greedy_matches_dp_on_atm_notes(self):
        for amount in range(0, 2000, 10):
            g, rest = cc.greedy_change(amount, ATM)
            self.assertEqual(rest, 0)
            self.assertEqual(cc.total_notes(g), cc.solve_dp(amount, ATM).min_count)

    def test_dp_never_worse_than_greedy(self):
        random.seed(3)
        for _ in range(300):
            denoms = random.sample(range(1, 30), random.randint(1, 5))
            amount = random.randint(0, 200)
            g, rest = cc.greedy_change(amount, denoms)
            r = cc.solve_dp(amount, denoms)
            if rest == 0:
                self.assertLessEqual(r.min_count, cc.total_notes(g))

    def test_matches_independent_bfs(self):
        random.seed(11)
        for _ in range(300):
            denoms = random.sample(range(2, 40), random.randint(1, 5))
            amount = random.randint(0, 150)
            self.assertEqual(cc.solve_dp(amount, denoms).min_count, bfs_min(amount, denoms))

    def test_recorded_steps_match_table(self):
        r = cc.solve_dp(60, ATM, record_steps=True)
        self.assertEqual(len(r.steps), 60)
        for s in r.steps:
            expect = None if r.dp[s.amount] == cc.INF else int(r.dp[s.amount])
            self.assertEqual(s.value, expect)

    def test_path_goes_from_amount_to_zero(self):
        r = cc.solve_dp(270, ATM)
        self.assertEqual(r.path()[0], 270)
        self.assertEqual(r.path()[-1], 0)
        self.assertEqual(len(r.path()), r.min_count + 1)

    def test_input_order_does_not_matter(self):
        self.assertEqual(cc.solve_dp(370, [10, 500, 50, 200, 100]).notes,
                         cc.solve_dp(370, ATM).notes)


class TestLimitedStock(unittest.TestCase):
    def test_stock_forces_more_notes(self):
        self.assertEqual(cc.solve_limited(1000, [500, 100], [1, 10]), {500: 1, 100: 5})

    def test_not_enough_stock(self):
        self.assertIsNone(cc.solve_limited(1000, [500, 100], [1, 3]))

    def test_unlimited_equivalent_when_stock_large(self):
        a = cc.solve_limited(3770, ATM, [100] * 5)
        self.assertEqual(sum(a.values()), cc.solve_dp(3770, ATM).min_count)

    def test_zero_stock_denomination_unused(self):
        r = cc.solve_limited(600, [500, 100], [0, 10])
        self.assertEqual(r, {100: 6})

    def test_amount_zero(self):
        self.assertEqual(cc.solve_limited(0, ATM, [1] * 5), {})

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            cc.solve_limited(100, [100, 50], [1])
        with self.assertRaises(ValueError):
            cc.solve_limited(100, [100, 50], [1, -1])

    def test_matches_brute_force(self):
        random.seed(5)
        for _ in range(200):
            k = random.randint(1, 3)
            denoms = random.sample(range(1, 15), k)
            stocks = [random.randint(0, 4) for _ in range(k)]
            amount = random.randint(0, 40)
            got = cc.solve_limited(amount, denoms, stocks)
            exp = brute_limited(amount, denoms, stocks)
            if exp is None:
                self.assertIsNone(got)
            else:
                self.assertEqual(sum(got.values()), exp)
                self.assertEqual(sum(d * n for d, n in got.items()), amount)
                for d, s in zip(denoms, stocks):
                    self.assertLessEqual(got.get(d, 0), s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
