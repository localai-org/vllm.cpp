import sys, statistics as st

rows = []
for line in open(sys.argv[1]):
    if '[FLINT-PHASE]' not in line:
        continue
    d = {}
    for kv in line.split()[2:]:
        k, v = kv.split('=')
        d[k] = float(v)
    rows.append(d)

decode = rows[1:]  # skip the prefill line
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 10
n = len(decode)
e = [r['attn_sliding'] for r in decode[lo:lo + 11]]
l = [r['attn_sliding'] for r in decode[-14:]]
ef = [r['attn_full'] for r in decode[lo:lo + 11]]
lf = [r['attn_full'] for r in decode[-14:]]
me, ml = st.median(e), st.median(l)
mef, mlf = st.median(ef), st.median(lf)
xs = list(range(n))
ys = [r['attn_sliding'] for r in decode]
mx = (n - 1) / 2
my = sum(ys) / n
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
wall = [r['fwd_wall_ms'] for r in decode]
print(f"steps={n} early_sliding={me:.3f} late_sliding={ml:.3f} ratio={ml / me:.3f} "
      f"early_full={mef:.3f} late_full={mlf:.3f} full_ratio={mlf / mef:.3f} "
      f"slope={slope:.4f} intercept={my - slope * mx:.3f} med_wall={st.median(wall):.3f}")
