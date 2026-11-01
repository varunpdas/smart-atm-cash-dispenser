"""
atm_gui.py
==========
Smart ATM Cash Dispenser - Dynamic Programming (Minimum Coin Change) visualizer.

Run:   python atm_gui.py
Needs: Python 3.8+ with Tkinter (bundled with the standard installer).
"""

import tkinter as tk
from tkinter import messagebox, ttk

import coin_core as cc

SYM = "\u20b9"            # rupee sign
INF_SYM = "\u221e"        # infinity sign
VIS_LIMIT = 300           # largest amount whose DP table is drawn step by step
COLS, CELL_W, CELL_H, PAD = 20, 58, 52, 20

PRESETS = {
    "ATM withdrawal (270)": ("270", "500, 200, 100, 50, 10", ""),
    "Greedy fails: notes 1,6,10 - amount 12": ("12", "1, 6, 10", ""),
    "Greedy fails: notes 1,10,25 - amount 30": ("30", "25, 10, 1", ""),
    "Greedy gets stuck: notes 4,5 - amount 8": ("8", "5, 4", ""),
    "Impossible amount: notes 10,20 - amount 35": ("35", "10, 20", ""),
    "Limited stock: 100 x1, 50 x3 - amount 250": ("250", "100, 50", "1, 3"),
    "Large amount (table not drawn): 3770": ("3770", "500, 200, 100, 50, 10", ""),
}

ALGO_INFO = """\
MINIMUM COIN / NOTE CHANGE  (Dynamic Programming)
==================================================

Problem
  An ATM holds notes of several denominations. Pay out an amount A using the
  FEWEST notes possible.

Why not just be greedy?
  "Always hand out the biggest note that fits" is fast, but it is NOT always
  optimal. With notes {1, 6, 10} and amount 12, greedy pays 10+1+1 (3 notes)
  while the best answer is 6+6 (2 notes). With notes {5, 4} and amount 8,
  greedy takes 5 and gets stuck with 3 left, yet 4+4 works. Real ATM notes
  (500, 200, 100, 50, 10) happen to be "greedy-friendly", but that is not
  true for every denomination set - so we need DP to be safe.

Optimal substructure
  If the best payout for A ends with a note d, the rest must be the best payout
  for A - d. So the answer is built from answers to smaller amounts.

Recurrence
  dp[0] = 0
  dp[a] = min over every note d <= a of ( dp[a - d] + 1 )
  dp[a] = infinity if no note leads to a payable smaller amount.
  choice[a] stores the note that gave the minimum.

Traceback
  Start at A, repeatedly take choice[a], subtract it, and stop at 0.
  The notes collected are the payout.

Time complexity   (A = amount, k = number of denominations)
  Fill the table ..... O(A x k)       (A cells, k candidates each)
  Traceback .......... O(number of notes) <= O(A)
  Greedy (compare) ... O(k)
  Plain recursion .... exponential (it recomputes the same sub-amounts)

Space complexity
  O(A) for dp[] and choice[].

Limited stock (optional)
  If the ATM has only s_i notes of denomination d_i, each stock is split into
  chunks 1,2,4,... and solved as a 0/1 knapsack over chunks:
  O(A x sum of log s_i) time, O(A x number of chunks) space for the traceback.

Tie-breaking here
  When two notes give the same count, the larger note is preferred.
"""


class ATMApp:
    def __init__(self, master):
        self.master = master
        master.title("Smart ATM Cash Dispenser - Dynamic Programming Visualizer")
        master.geometry("1280x800")
        master.minsize(1000, 660)

        self.res = None            # DPResult
        self.greedy = ({}, 0)      # (counts, remainder)
        self.limited = None        # dict / None
        self.use_stock = False
        self.visual = False
        self.idx = 0
        self.total = 0
        self.playing = False
        self.after_id = None

        self._build_ui()
        self.apply_preset(next(iter(PRESETS)))
        self._update_controls()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        paned = ttk.PanedWindow(self.master, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, padding=8)
        right = ttk.Frame(paned, padding=8)
        paned.add(left, weight=1)
        paned.add(right, weight=4)

        # ---- left: inputs
        box = ttk.LabelFrame(left, text="1. Withdrawal details", padding=6)
        box.pack(fill="x")
        ttk.Label(box, text="Preset example:").pack(anchor="w")
        self.cmb = ttk.Combobox(box, values=list(PRESETS), state="readonly", width=38)
        self.cmb.pack(fill="x", pady=(0, 6))
        self.cmb.bind("<<ComboboxSelected>>", lambda e: self.apply_preset(self.cmb.get()))
        ttk.Label(box, text=f"Amount to withdraw ({SYM}):").pack(anchor="w")
        self.e_amount = ttk.Entry(box)
        self.e_amount.pack(fill="x", pady=(0, 6))
        ttk.Label(box, text="Note denominations (comma separated):").pack(anchor="w")
        self.e_denoms = ttk.Entry(box)
        self.e_denoms.pack(fill="x", pady=(0, 6))
        ttk.Label(box, text="Stock per note, same order (optional):").pack(anchor="w")
        self.e_stock = ttk.Entry(box)
        self.e_stock.pack(fill="x")

        # ---- left: controls
        ctl = ttk.LabelFrame(left, text="2. Algorithm controls", padding=6)
        ctl.pack(fill="x", pady=8)
        ttk.Button(ctl, text="Solve (build DP table)", command=self.solve).pack(fill="x")
        r2 = ttk.Frame(ctl)
        r2.pack(fill="x", pady=(6, 0))
        self.btn_prev = ttk.Button(r2, text="< Prev", width=8, command=self.prev_step)
        self.btn_next = ttk.Button(r2, text="Next >", width=8, command=self.next_step)
        self.btn_play = ttk.Button(r2, text="Auto-play", width=9, command=self.toggle_play)
        self.btn_end = ttk.Button(r2, text="Finish", width=7, command=self.finish)
        for b in (self.btn_prev, self.btn_next, self.btn_play, self.btn_end):
            b.pack(side="left", padx=2)
        r3 = ttk.Frame(ctl)
        r3.pack(fill="x", pady=(6, 0))
        ttk.Label(r3, text="Delay (ms):").pack(side="left")
        self.speed = tk.Scale(r3, from_=100, to=1500, orient="horizontal",
                              showvalue=True, resolution=50)
        self.speed.set(500)
        self.speed.pack(side="left", fill="x", expand=True, padx=4)
        self.btn_reset = ttk.Button(ctl, text="Reset to step 0", command=self.reset)
        self.btn_reset.pack(fill="x", pady=(6, 0))

        # ---- left: legend
        leg = ttk.LabelFrame(left, text="Legend", padding=6)
        leg.pack(fill="x")
        ttk.Label(leg, justify="left", text=(
            "Each cell = one amount\n"
            "small number : amount\n"
            "big number   : fewest notes (dp)\n"
            f"{INF_SYM}            : cannot be paid\n"
            "Gold fill    : cell being computed\n"
            "Orange edge  : cells it looks back at\n"
            "Red edge     : the best choice\n"
            "Blue fill    : final payout path"),
            font=("Courier", 9)).pack(anchor="w")

        # ---- right: step info
        self.lbl_step = ttk.Label(right, text="Press 'Solve' to start.",
                                  font=("TkDefaultFont", 11, "bold"))
        self.lbl_step.pack(anchor="w")
        self.lbl_msg = ttk.Label(right, text="", wraplength=900, justify="left")
        self.lbl_msg.pack(anchor="w", pady=(2, 4))

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        tab_vis = ttk.Frame(nb)
        tab_info = ttk.Frame(nb)
        nb.add(tab_vis, text="DP table")
        nb.add(tab_info, text="Algorithm & complexity")

        self.canvas = tk.Canvas(tab_vis, bg="white", highlightthickness=0)
        hs = ttk.Scrollbar(tab_vis, orient="horizontal", command=self.canvas.xview)
        vs = ttk.Scrollbar(tab_vis, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hs.set, yscrollcommand=vs.set)
        hs.pack(side="bottom", fill="x")
        vs.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        info = tk.Text(tab_info, wrap="word", font=("Courier", 10), padx=8, pady=8)
        info.insert("1.0", ALGO_INFO)
        info.configure(state="disabled")
        info.pack(fill="both", expand=True)

        # ---- right: details of the current cell
        det = ttk.LabelFrame(right, text="Current cell: candidates considered", padding=6)
        det.pack(fill="x", pady=(8, 0))
        self.txt_detail = tk.Text(det, height=6, font=("Courier", 10), wrap="none")
        self.txt_detail.pack(fill="x")
        self.txt_detail.configure(state="disabled")

        # ---- right: results
        res = ttk.LabelFrame(right, text="Dispense result", padding=6)
        res.pack(fill="x", pady=(8, 0))
        self.lbl_result = ttk.Label(res, text="(available when the table is complete)",
                                    justify="left", font=("Courier", 10))
        self.lbl_result.pack(anchor="w")

    # ------------------------------------------------------------- presets
    def apply_preset(self, name):
        amount, denoms, stock = PRESETS[name]
        self.cmb.set(name)
        for entry, value in ((self.e_amount, amount), (self.e_denoms, denoms),
                             (self.e_stock, stock)):
            entry.delete(0, "end")
            entry.insert(0, value)

    # --------------------------------------------------------------- solve
    def solve(self):
        self._stop_play()
        try:
            amounts = cc.parse_int_list(self.e_amount.get())
            if len(amounts) != 1:
                raise ValueError("Enter a single amount.")
            amount = amounts[0]
            denoms = cc.parse_int_list(self.e_denoms.get())
            stock_text = self.e_stock.get().strip()
            stocks = cc.parse_int_list(stock_text) if stock_text else None
            visual = 0 <= amount <= VIS_LIMIT
            res = cc.solve_dp(amount, denoms, record_steps=visual)
            greedy = cc.greedy_change(amount, denoms)
            limited = (cc.solve_limited(amount, denoms, stocks)
                       if stocks is not None else None)
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        self.res, self.greedy, self.limited = res, greedy, limited
        self.use_stock = stocks is not None
        self.visual = visual
        self.total = amount if visual else 0
        self.idx = 0
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self._show_step()

    # ---------------------------------------------------------- show a step
    def _is_final(self):
        return (not self.visual) or self.idx == self.total

    def _show_step(self):
        final = self._is_final()
        if self.visual:
            self.lbl_step.configure(
                text=f"Step {self.idx} of {self.total}" + ("  -  table complete" if final else ""))
            self.lbl_msg.configure(text=self._step_message())
        else:
            self.lbl_step.configure(text="Large amount: result only")
            self.lbl_msg.configure(
                text=f"Amounts above {VIS_LIMIT} are solved instantly but the table is not "
                     f"drawn (it would be too large). Use a smaller amount to watch the steps.")
        self._set_detail(self._detail_text())
        self._draw()
        if final:
            self._show_result()
        else:
            self.lbl_result.configure(text="(available when the table is complete)")
        self._update_controls()

    def _step_message(self):
        a = self.idx
        if a == 0:
            return "Base case: dp[0] = 0, because paying amount 0 needs zero notes."
        step = self.res.steps[a - 1]
        if step.value is None:
            return (f"Amount {a}: no note fits into a payable smaller amount, "
                    f"so dp[{a}] = {INF_SYM} (cannot be paid).")
        return (f"Amount {a}: best is note {SYM}{step.used}, giving dp[{a}] = "
                f"dp[{a - step.used}] + 1 = {step.value}.")

    def _detail_text(self):
        if not self.visual or self.idx == 0:
            return "Step 0: dp[0] = 0 (base case). Press Next to compute dp[1], dp[2], ..."
        a = self.idx
        step = self.res.steps[a - 1]
        lines = [f"dp[{a}] = min over notes d <= {a} of ( dp[{a} - d] + 1 )"]
        if not step.candidates:
            lines.append(f"  no note is <= {a}")
        for d, val in step.candidates:
            shown = "unreachable" if val is None else str(val)
            mark = "   <-- best" if d == step.used and val is not None else ""
            lines.append(f"  note {SYM}{d:<5} dp[{a - d}] + 1 = {shown}{mark}")
        lines.append(f"=> dp[{a}] = {INF_SYM if step.value is None else step.value}")
        return "\n".join(lines)

    def _set_detail(self, text):
        self.txt_detail.configure(state="normal")
        self.txt_detail.delete("1.0", "end")
        self.txt_detail.insert("1.0", text)
        self.txt_detail.configure(state="disabled")

    # ------------------------------------------------------------- drawing
    def _draw(self):
        c = self.canvas
        c.delete("all")
        if not self.visual:
            c.create_text(30, 30, anchor="nw", fill="#555", font=("TkDefaultFont", 11),
                          text="DP table not drawn for this amount.\nSee the result below.")
            c.configure(scrollregion=(0, 0, 400, 100))
            return
        res, idx, final = self.res, self.idx, self._is_final()
        sources, chosen = set(), None
        if idx >= 1:
            step = res.steps[idx - 1]
            sources = {idx - d for d, _ in step.candidates}
            if step.used:
                chosen = idx - step.used
        path = set(res.path()) if final else set()
        bold = ("TkDefaultFont", 11, "bold")
        tiny = ("TkDefaultFont", 7)

        for a in range(res.amount + 1):
            r, col = divmod(a, COLS)
            x, y = PAD + col * CELL_W, PAD + r * CELL_H
            computed = a <= idx
            value = res.dp[a]
            if a == idx:
                fill = "#f9e79f"
            elif a in path:
                fill = "#aed6f1"
            elif computed:
                fill = "#d5f5e3" if value != cc.INF else "#e5e7e9"
            else:
                fill = "white"
            outline, width = "#95a5a6", 1
            if a in sources and a != idx:
                outline, width = "#e67e22", 3
            if a == chosen:
                outline, width = "#c0392b", 3
            c.create_rectangle(x, y, x + CELL_W - 4, y + CELL_H - 4, fill=fill,
                               outline=outline, width=width)
            c.create_text(x + 4, y + 3, anchor="nw", text=str(a), font=tiny, fill="#7f8c8d")
            if computed:
                c.create_text(x + (CELL_W - 4) / 2, y + (CELL_H - 4) / 2 + 6,
                              text=INF_SYM if value == cc.INF else str(int(value)),
                              font=bold)

        rows = res.amount // COLS + 1
        total_h = PAD * 2 + rows * CELL_H
        c.configure(scrollregion=(0, 0, PAD * 2 + COLS * CELL_W, total_h))
        cur_row = idx // COLS
        c.yview_moveto(max(0.0, (cur_row * CELL_H - 2 * CELL_H) / total_h))

    # ------------------------------------------------------------- results
    def _show_result(self):
        res = self.res
        g_counts, g_rest = self.greedy
        dp_notes = res.notes
        lines = [
            f"Amount         : {SYM}{res.amount}",
            f"Denominations  : {', '.join(SYM + str(d) for d in res.denoms)}",
        ]
        if dp_notes is None:
            lines.append("DP (optimal)   : cannot be paid with these notes")
        else:
            lines.append(f"DP (optimal)   : {cc.format_notes(dp_notes, SYM)}"
                         f"   -> {res.min_count} notes")
        if g_rest:
            lines.append(f"Greedy         : gets stuck, {SYM}{g_rest} left over")
        else:
            lines.append(f"Greedy         : {cc.format_notes(g_counts, SYM)}"
                         f"   -> {cc.total_notes(g_counts)} notes")
        lines.append(f"Verdict        : {self._verdict()}")
        if self.use_stock:
            if self.limited is None:
                lines.append("With stock     : cannot be paid from the available stock")
            else:
                lines.append(f"With stock     : {cc.format_notes(self.limited, SYM)}"
                             f"   -> {cc.total_notes(self.limited)} notes")
        else:
            lines.append("With stock     : (no stock limits entered)")
        self.lbl_result.configure(text="\n".join(lines))

    def _verdict(self):
        res = self.res
        g_counts, g_rest = self.greedy
        if res.min_count is None:
            return "no combination of these notes can make this amount."
        if g_rest:
            return "greedy fails to pay this amount, but DP finds a payout."
        g_n = cc.total_notes(g_counts)
        if g_n > res.min_count:
            return f"greedy is NOT optimal here ({g_n} notes vs {res.min_count} with DP)."
        return "greedy happens to be optimal for this input."

    # ----------------------------------------------------------- navigation
    def next_step(self):
        if self.visual and self.idx < self.total:
            self.idx += 1
            self._show_step()

    def prev_step(self):
        if self.visual and self.idx > 0:
            self.idx -= 1
            self._show_step()

    def finish(self):
        self._stop_play()
        if self.visual:
            self.idx = self.total
            self._show_step()

    def reset(self):
        self._stop_play()
        if self.visual:
            self.idx = 0
            self._show_step()

    def toggle_play(self):
        if self.playing:
            self._stop_play()
            return
        if not self.visual:
            return
        if self.idx == self.total:
            self.idx = 0
            self._show_step()
        self.playing = True
        self.btn_play.configure(text="Pause")
        self._tick()

    def _tick(self):
        if not self.playing:
            return
        if self.idx >= self.total:
            self._stop_play()
            return
        self.next_step()
        self.after_id = self.master.after(int(self.speed.get()), self._tick)

    def _stop_play(self):
        self.playing = False
        if self.after_id is not None:
            self.master.after_cancel(self.after_id)
            self.after_id = None
        self.btn_play.configure(text="Auto-play")

    def _update_controls(self):
        have = self.res is not None and self.visual
        last = have and self.idx == self.total
        state = lambda cond: "normal" if cond else "disabled"
        self.btn_prev.configure(state=state(have and self.idx > 0))
        self.btn_next.configure(state=state(have and not last))
        self.btn_end.configure(state=state(have and not last))
        self.btn_reset.configure(state=state(have and self.idx > 0))
        self.btn_play.configure(state=state(have))


def main():
    root = tk.Tk()
    ATMApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
