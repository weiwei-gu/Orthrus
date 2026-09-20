"""对比两个推力恢复测试结果版本"""
import json
import sys

import numpy as np


def load(path):
    return [t for t in json.load(open(path)) if t["angle_deg"] >= 0]


def summarize(trials):
    out = {}
    for cond in ("trot", "walk"):
        for mag in (10.0, 20.0, 30.0):
            sel = [t for t in trials if t["cond"] == cond and t["mag"] == mag]
            n_pass = sum(t["pass"] for t in sel)
            recs = [t["recover_s"] for t in sel if t["recover_s"] is not None]
            out[(cond, mag)] = (n_pass, len(sel),
                                np.mean(recs) if recs else float("nan"))
    return out


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    A, B = summarize(load(a_path)), summarize(load(b_path))
    print(f"{'工况':<6}{'冲量':>8}   {a_path.split('/')[-1]:>14}   {b_path.split('/')[-1]:>14}")
    for key in sorted(A):
        cond, mag = key
        pa, na, ra = A[key]
        pb, nb, rb = B[key]
        print(f"{cond:<6}{mag:>5.0f}N·s   {pa}/{na} ({100*pa/na:4.0f}%) rec {ra:4.1f}s"
              f"   {pb}/{nb} ({100*pb/nb:4.0f}%) rec {rb:4.1f}s")
    tot_a = sum(t['pass'] for t in load(a_path))
    tot_b = sum(t['pass'] for t in load(b_path))
    n = len(load(a_path))
    print(f"\n总计: {a_path.split('/')[-1]} {tot_a}/{n}  ->  "
          f"{b_path.split('/')[-1]} {tot_b}/{n}")


if __name__ == "__main__":
    main()
