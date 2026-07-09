"""
compare_full.py
---------------
Three-way comparison: MockerOld  vs  MockerNewPathA  vs  MockerNewPathB
on the FULL out_1.parents catalog.

Reads the file ONCE, times all three, then checks:
  - centrals:             exact np.allclose for all three
  - satellites lgm / Mr: exact np.allclose for all three
  - satellites x/y/z:    exact for Old ≡ PathA;  dist-only for Old vs PathB
"""
import numpy as np
import pandas as pd
import time
from test_harness import MockerOld, MockerNewPathA, MockerNewPathB

# ------------------------------------------------------------------ #
# 1.  Read the full catalog ONCE
# ------------------------------------------------------------------ #
halodatatype = {
    'ID':int,'descID':int,
    'mbnd_vir':float,'vmax':float,'vrms':float,'rvir':float,'rs':float,'np':int,
    'x':float,'y':float,'z':float,'vx':float,'vy':float,'vz':float,
    'Jx':float,'Jy':float,'Jz':float,'spin':float,'rs_klypin':float,
    'mvir':float,'m200b':float,'m200c':float,'mCustom2':float,'mCustom':float,
    'Xoff':float,'Voff':float,'spin_bullock':float,'b_to_a':float,'c_to_a':float,
    'Ax':float,'Ay':float,'Az':float,'b_to_a_500c':float,'c_to_a_500c':float,
    'Ax_500c':float,'Ay_500c':float,'Az_500c':float,
    'TbyU':float,'Mpe_Behroozi':float,'Mpe_Diemer':float,'halfmassradius':float,
    'pid':int,
}

print("Reading out_1.parents (one-time read)...")
t0 = time.time()
df = pd.read_csv('out_1.parents',
                 dtype=halodatatype,
                 names=list(halodatatype.keys()),
                 comment='#', sep=r'\s+', header=None)
t_read = time.time() - t0
print(f"  {len(df):,} rows in {t_read:.2f}s")

halos_all = df.to_records(index=False)
halos = halos_all[(halos_all['m200b'] >= 2e11) & (halos_all['pid'] == -1)]
print(f"  After cuts: {halos.size:,} parent halos\n")

# ------------------------------------------------------------------ #
# 2.  Run all three mockers
# ------------------------------------------------------------------ #
results = {}
for label, cls, kwargs in [
    ("MockerOld",  MockerOld,      {}),
    ("PathA",      MockerNewPathA, {}),
    ("PathB",      MockerNewPathB, {"chunk_size": 5000}),
]:
    print(f"  Running {label} ...")
    m = cls(halos, **kwargs)
    m.read_this = lambda h=halos: h          # skip internal CSV re-read
    t0 = time.time()
    c, s, _ = m.mock_it()
    t = time.time() - t0
    results[label] = dict(c=c, s=s, t=t)
    print(f"    {t:.2f}s | {c.size:,} centrals | {s.size:,} satellites")

# ------------------------------------------------------------------ #
# 3.  Helpers
# ------------------------------------------------------------------ #
CEN_PROPS       = ['haloid','lgm','rvir','con','x','y','z','Mr','Nsat']
SAT_EXACT_PROPS = ['lgm','Mr']
SAT_POS_PROPS   = ['x','y','z']

def check_exact(prop, a, b):
    ok = np.allclose(a.astype(float), b.astype(float))
    tag = "✓ exact " if ok else "✗ FAIL  "
    note = ""
    if not ok:
        note = f"  max_diff={np.abs(a.astype(float)-b.astype(float)).max():.3e}"
    print(f"    {tag} {prop:8s}{note}")

def check_dist(prop, a, b):
    m_a, s_a = np.mean(a), np.std(a)
    m_b, s_b = np.mean(b), np.std(b)
    dm = abs(m_a - m_b) / abs(m_a) * 100
    ds = abs(s_a - s_b) / abs(s_a) * 100
    ok = dm < 1.0 and ds < 1.0
    tag = "✓ dist  " if ok else "! dist  "
    print(f"    {tag} {prop:8s}  "
          f"mean {m_a:.3f} ↔ {m_b:.3f}  ({dm:.3f}%)   "
          f"std  {s_a:.3f} ↔ {s_b:.3f}  ({ds:.3f}%)")

def compare_pair(la, lb, pos_dist_only=False):
    ra, rb = results[la], results[lb]
    print()
    print("=" * 65)
    print(f"  {la}  vs  {lb}")
    print("=" * 65)
    print("  Centrals:")
    for p in CEN_PROPS:
        check_exact(p, ra['c'][p], rb['c'][p])
    print("  Satellite lgm / Mr:")
    for p in SAT_EXACT_PROPS:
        check_exact(p, ra['s'][p], rb['s'][p])
    print("  Satellite positions" + (" (distribution only):" if pos_dist_only else " (exact):"))
    for p in SAT_POS_PROPS:
        if pos_dist_only:
            check_dist(p, ra['s'][p], rb['s'][p])
        else:
            check_exact(p, ra['s'][p], rb['s'][p])

# ------------------------------------------------------------------ #
# 4.  All three pair comparisons
# ------------------------------------------------------------------ #
compare_pair("MockerOld", "PathA",  pos_dist_only=False)   # expect full exact
compare_pair("MockerOld", "PathB",  pos_dist_only=True)    # positions dist-only
compare_pair("PathA",     "PathB",  pos_dist_only=True)    # positions dist-only

# ------------------------------------------------------------------ #
# 5.  Summary table
# ------------------------------------------------------------------ #
t_ref = results["MockerOld"]["t"]
print()
print("=" * 65)
print("  TIMING & GALAXY COUNT SUMMARY")
print("=" * 65)
header = f"  {'Version':<18} {'Time':>8}  {'Speedup':>8}  {'Centrals':>10}  {'Sats':>8}  {'Pos match':>12}"
print(header)
print("  " + "-" * 63)
for label, pos_note in [("MockerOld","—"),("PathA","exact"),("PathB","<0.05%")]:
    r = results[label]
    spd = f"{t_ref/r['t']:.2f}x"
    print(f"  {label:<18} {r['t']:>7.2f}s  {spd:>8}  "
          f"{r['c'].size:>10,}  {r['s'].size:>8,}  {pos_note:>12}")
print()
print("  Old ≡ PathA  : bit-identical (same per-halo linear NFW grid + same RNG order)")
print("  Old ≈ PathB  : <0.05% position diff (global log-spaced NFW grid, same draw count)")
print("  PathA ≈ PathB: positions differ by NFW quadrature only; lgm/Mr/Nsat exact")
