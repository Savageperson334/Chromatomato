import os
import math
import traceback
from bisect import bisect_left, bisect_right

import matplotlib
import matplotlib.pyplot as plt

# =========================
# EDIT THIS
# =========================
INPUT_FILE = r"1.TXT"          # put your Compass TXT path here
SNAP_Y_MODE = "nearest"        # "nearest" or "same_peak"
NODE_SIZE = 90
HIGHLIGHT_SIZE = 320
BASELINE_LW = 2.5
PRINT_AREA_CHECK_ON_START = True
ALWAYS_SELECT_NEAREST = True   # TRUE = ignore radius; always select closest node on click
MAX_SELECT_RADIUS_PX = 200     # used only if ALWAYS_SELECT_NEAREST = False
DEBUG_EVERY_CLICK = True       # prints a line on every click so we know handler runs
# =========================

# IMPORTANT:
# Matplotlib stores callbacks to bound methods via weakrefs.
# If the editor object is not kept alive, the callbacks can silently stop working.
_KEEP_ALIVE = None


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def load_compass_txt(path: str):
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    # raw trace
    times, values = [], []
    i = 0
    while i < len(lines) and not lines[i].startswith("Time\tValue"):
        i += 1
    if i >= len(lines):
        raise RuntimeError("Could not find 'Time\\tValue' header in TXT.")
    i += 1
    while i < len(lines) and not lines[i].startswith("Index\t"):
        parts = lines[i].strip().split("\t")
        if len(parts) >= 2 and is_number(parts[0]) and is_number(parts[1]):
            times.append(float(parts[0]))
            values.append(float(parts[1]))
        i += 1

    if i >= len(lines):
        return times, values, []

    # peak table header
    header_cols = lines[i].rstrip("\n").split("\t")
    col_indices = {}
    for idx, name in enumerate(header_cols):
        col_indices.setdefault(name, []).append(idx)

    def pick_col(name: str, which="first"):
        if name not in col_indices:
            return None
        return col_indices[name][-1] if which == "last" else col_indices[name][0]

    idx_name = pick_col("Name", "first")
    idx_rt = pick_col("Time", "first")
    idx_bs_t = pick_col("Baseline Start Time", "last")
    idx_bs_v = pick_col("Baseline Start Value", "last")
    idx_be_t = pick_col("Baseline Stop Time", "first")
    idx_be_v = pick_col("Baseline Stop Value", "first")
    idx_area = col_indices["Area"][-1] if "Area" in col_indices else None

    needed = [idx_name, idx_rt, idx_bs_t, idx_bs_v, idx_be_t, idx_be_v]
    if any(x is None for x in needed):
        return times, values, []

    i += 2  # skip units row

    peaks = []
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if not line.strip():
            i += 1
            continue
        if line.startswith("Total") or line.startswith("Totals"):
            break

        cols = line.split("\t")
        if len(cols) <= max(needed):
            i += 1
            continue

        name = cols[idx_name].strip()
        rt_s = cols[idx_rt].strip()
        bs_t_s = cols[idx_bs_t].strip()
        bs_v_s = cols[idx_bs_v].strip()
        be_t_s = cols[idx_be_t].strip()
        be_v_s = cols[idx_be_v].strip()

        if not (is_number(rt_s) and is_number(bs_t_s) and is_number(bs_v_s) and is_number(be_t_s) and is_number(be_v_s)):
            i += 1
            continue

        exp_area = None
        if idx_area is not None and idx_area < len(cols):
            a_s = cols[idx_area].strip()
            if is_number(a_s):
                exp_area = float(a_s)

        peaks.append({
            "name": name,
            "rt": float(rt_s),
            "bs_t": float(bs_t_s),
            "bs_v": float(bs_v_s),
            "be_t": float(be_t_s),
            "be_v": float(be_v_s),
            "export_area": exp_area,
        })
        i += 1

    return times, values, peaks


def baseline_y_at_x(p, x):
    bs_t, bs_v = p["bs_t"], p["bs_v"]
    be_t, be_v = p["be_t"], p["be_v"]
    if be_t == bs_t:
        return bs_v
    m = (be_v - bs_v) / (be_t - bs_t)
    return bs_v + m * (x - bs_t)


def compute_area_uv_sec(times, values, peak):
    bs_t, be_t = peak["bs_t"], peak["be_t"]
    bs_v, be_v = peak["bs_v"], peak["be_v"]
    if not times or be_t <= bs_t:
        return 0.0

    i0 = bisect_left(times, bs_t)
    i1 = bisect_right(times, be_t) - 1
    if i1 <= i0:
        return 0.0

    def b(t):
        if be_t == bs_t:
            return bs_v
        m = (be_v - bs_v) / (be_t - bs_t)
        return bs_v + m * (t - bs_t)

    area_uv_min = 0.0
    t_prev = times[i0]
    y_prev = values[i0] - b(t_prev)
    for k in range(i0 + 1, i1 + 1):
        t = times[k]
        y = values[k] - b(t)
        area_uv_min += 0.5 * (y_prev + y) * (t - t_prev)
        t_prev, y_prev = t, y
    return area_uv_min * 60.0


def print_area_check(times, values, peaks):
    print(f"Raw points: {len(times)}")
    print(f"Peaks parsed: {len(peaks)}\n")
    print("=== Area check (Baseline Start/Stop; µV*sec) ===\n")
    for p in sorted(peaks, key=lambda x: x["rt"]):
        computed = compute_area_uv_sec(times, values, p)
        exp = p.get("export_area")

        print(f"Peak: {p['name']}  RT={p['rt']:.4f} min")
        print(f"  Baseline: {p['bs_t']:.4f},{p['bs_v']:.3f}  ->  {p['be_t']:.4f},{p['be_v']:.3f}")
        if exp is not None and exp != 0:
            diff = computed - exp
            pct = diff / exp * 100.0
            print(f"  Exported Area: {exp:.3f}")
            print(f"  Computed Area: {computed:.3f}")
            print(f"  Diff: {diff:+.3f} ({pct:+.2f}%)\n")
        else:
            print(f"  Computed Area: {computed:.3f}\n")


class ClickSelectPlaceEditor:
    def __init__(self, times, values, peaks):
        self.times = times
        self.values = values
        self.peaks = sorted(peaks, key=lambda p: p["rt"])

        self.fig, self.ax = plt.subplots(figsize=(13, 6))
        self.fig.canvas.manager.set_window_title("Chromatogram Editor (select → click place)")

        self.ax.plot(times, values, linewidth=1)

        self.baseline_lines = []
        for p in self.peaks:
            (ln,) = self.ax.plot([p["bs_t"], p["be_t"]], [p["bs_v"], p["be_v"]],
                                 linewidth=BASELINE_LW, alpha=0.9, zorder=1)
            self.baseline_lines.append(ln)
            self.ax.axvline(p["rt"], linestyle="--", alpha=0.25, zorder=0)

        self.node_x, self.node_y, self.node_meta = [], [], []
        for pi, p in enumerate(self.peaks):
            self.node_x += [p["bs_t"], p["be_t"]]
            self.node_y += [p["bs_v"], p["be_v"]]
            self.node_meta += [(pi, "bs"), (pi, "be")]

        self.nodes = self.ax.scatter(self.node_x, self.node_y, s=NODE_SIZE, zorder=5)

        self.status = self.ax.text(
            0.01, 0.99,
            "Left click: select nearest node (then place). Right click/ESC: clear.",
            transform=self.ax.transAxes, va="top", ha="left",
            fontsize=10, bbox=dict(boxstyle="round", alpha=0.12)
        )

        self.selected_node = None

        cid1 = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        cid2 = self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        print("Connected:", "button_press_event cid=", cid1, "key_press_event cid=", cid2)

        xs = self.node_x
        ys = self.node_y
        print(f"Nodes: {len(xs)}  X-range: {min(xs):.4f}..{max(xs):.4f}  Y-range: {min(ys):.3f}..{max(ys):.3f}")

        # Also pin self onto the figure to prevent GC (extra belt-and-suspenders)
        self.fig._chrom_editor = self

    def _highlight(self, idx):
        sizes = [NODE_SIZE] * len(self.node_x)
        if idx is not None:
            sizes[idx] = HIGHLIGHT_SIZE
        self.nodes.set_sizes(sizes)

    def _nearest_node_index(self, event):
        mx, my = event.x, event.y
        best_i = None
        best_d2 = float("inf")
        for i, (x, y) in enumerate(zip(self.node_x, self.node_y)):
            sx, sy = self.ax.transData.transform((x, y))
            d2 = (sx - mx) ** 2 + (sy - my) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        dist = math.sqrt(best_d2) if best_d2 != float("inf") else float("inf")
        return best_i, dist

    def _snap_y(self, x, mouse_y, peak_index):
        if SNAP_Y_MODE == "same_peak" and peak_index is not None:
            return baseline_y_at_x(self.peaks[peak_index], x)

        best_y, best_d = None, float("inf")
        for p in self.peaks:
            lo, hi = (p["bs_t"], p["be_t"]) if p["bs_t"] <= p["be_t"] else (p["be_t"], p["bs_t"])
            if not (lo <= x <= hi):
                continue
            yb = baseline_y_at_x(p, x)
            d = abs(yb - mouse_y) if mouse_y is not None else 0.0
            if d < best_d:
                best_d = d
                best_y = yb

        if best_y is None and peak_index is not None:
            return baseline_y_at_x(self.peaks[peak_index], x)
        return best_y if best_y is not None else (mouse_y if mouse_y is not None else 0.0)

    def _apply_node(self, node_i, x, mouse_y):
        pi, which = self.node_meta[node_i]
        p = self.peaks[pi]
        y = self._snap_y(x, mouse_y, pi)

        if which == "bs":
            p["bs_t"], p["bs_v"] = x, y
        else:
            p["be_t"], p["be_v"] = x, y

        if p["be_t"] < p["bs_t"]:
            p["bs_t"], p["be_t"] = p["be_t"], p["bs_t"]
            p["bs_v"], p["be_v"] = p["be_v"], p["bs_v"]

        for j, (pidx, w) in enumerate(self.node_meta):
            if pidx != pi:
                continue
            if w == "bs":
                self.node_x[j], self.node_y[j] = p["bs_t"], p["bs_v"]
            else:
                self.node_x[j], self.node_y[j] = p["be_t"], p["be_v"]

        self.nodes.set_offsets(list(zip(self.node_x, self.node_y)))
        self.baseline_lines[pi].set_data([p["bs_t"], p["be_t"]], [p["bs_v"], p["be_v"]])

        computed = compute_area_uv_sec(self.times, self.values, p)
        exp = p.get("export_area")
        if exp is not None and exp != 0:
            diff = computed - exp
            pct = diff / exp * 100.0
            self.status.set_text(
                f"{p['name']}  Export={exp:.3f}  Computed={computed:.3f}  Diff={diff:+.3f} ({pct:+.2f}%)"
            )
        else:
            self.status.set_text(f"{p['name']}  Computed={computed:.3f}")

    def on_click(self, event):
        try:
            if DEBUG_EVERY_CLICK:
                print("CLICK:", "button=", event.button, "inaxes=", (event.inaxes is self.ax),
                      "xdata=", event.xdata, "ydata=", event.ydata)

            if event.inaxes is not self.ax:
                return

            if event.button == 3:
                self.selected_node = None
                self._highlight(None)
                self.status.set_text("Cleared selection.")
                self.fig.canvas.draw_idle()
                return

            if event.button != 1:
                return

            if self.selected_node is None:
                idx, dist = self._nearest_node_index(event)
                if idx is None:
                    return
                if (not ALWAYS_SELECT_NEAREST) and dist > MAX_SELECT_RADIUS_PX:
                    self.status.set_text(f"No node within {MAX_SELECT_RADIUS_PX}px (nearest was {dist:.1f}px).")
                    self.fig.canvas.draw_idle()
                    return

                self.selected_node = idx
                self._highlight(idx)
                pi, which = self.node_meta[idx]
                p = self.peaks[pi]
                self.status.set_text(
                    f"Selected {which.upper()} for {p['name']} (nearest {dist:.1f}px). Now click to place."
                )
                self.fig.canvas.draw_idle()
                return

            if event.xdata is None:
                return
            self._apply_node(self.selected_node, float(event.xdata), event.ydata)
            self._highlight(self.selected_node)
            self.fig.canvas.draw_idle()
        except Exception:
            print("ERROR in on_click:")
            traceback.print_exc()

    def on_key(self, event):
        if not event or not getattr(event, "key", None):
            return
        if str(event.key).lower() == "escape":
            self.selected_node = None
            self._highlight(None)
            self.status.set_text("Cleared selection (ESC).")
            self.fig.canvas.draw_idle()


def main():
    global _KEEP_ALIVE
    print("Backend:", matplotlib.get_backend())
    print("RUNNING:", __file__)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(INPUT_FILE)

    times, values, peaks = load_compass_txt(INPUT_FILE)
    if not peaks:
        raise RuntimeError("No peaks parsed. Ensure TXT contains peak table with baseline columns.")

    if PRINT_AREA_CHECK_ON_START:
        print_area_check(times, values, peaks)

    # KEEP A STRONG REFERENCE
    _KEEP_ALIVE = ClickSelectPlaceEditor(times, values, peaks)

    plt.show()


if __name__ == "__main__":
    main()
