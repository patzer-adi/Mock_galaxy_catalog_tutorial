import numpy as np
import pandas as pd
import time
from test_harness import MockerOld, MockerNewPathA, MockerNewPathB

if __name__ == '__main__':
    halodatatype = {'ID':int,'descID':int,
                    'mbnd_vir':float,'vmax':float,'vrms':float,'rvir':float,'rs':float,'np':int,
                    'x':float,'y':float,'z':float,'vx':float,'vy':float,'vz':float,
                    'Jx':float,'Jy':float,'Jz':float,'spin':float,'rs_klypin':float,
                    'mvir':float,'m200b':float,'m200c':float,'mCustom2':float,'mCustom':float,
                    'Xoff':float,'Voff':float,'spin_bullock':float,'b_to_a':float,'c_to_a':float,
                    'Ax':float,'Ay':float,'Az':float,'b_to_a_500c':float,'c_to_a_500c':float,
                    'Ax_500c':float,'Ay_500c':float,'Az_500c':float,
                    'TbyU':float,'Mpe_Behroozi':float,'Mpe_Diemer':float,'halfmassradius':float,
                    'pid':int}

    print("Reading out_1.parents...")
    df = pd.read_csv('out_1.parents', 
                     dtype=halodatatype, 
                     names=list(halodatatype.keys()), 
                     comment='#', 
                     sep=r'\s+', 
                     header=None)
    
    # Filter halos
    df_clean = df[(df['m200b'] >= 2e11) & (df['pid'] == -1)]
    print(f"Total clean parent halos in catalog: {len(df_clean)}")
    
    # Sample 5,000 random parent halos
    print("Sampling 5,000 random halos...")
    df_sample = df_clean.sample(5000, random_state=0)
    halos_sample = df_sample.to_records(index=False)
    
    print("\n--- Running Baseline MockerOld on subsample ---")
    m_old = MockerOld(halos_sample)
    t0 = time.time()
    centrals_old, satellites_old, _ = m_old.mock_it()
    t_old = time.time() - t0
    print(f"MockerOld took {t_old:.4f}s. Centrals: {centrals_old.size}, Satellites: {satellites_old.size}")

    print("\n--- Running Vectorized MockerNewPathA (Exact Match) ---")
    m_new_a = MockerNewPathA(halos_sample, chunk_size=5000)
    t0 = time.time()
    centrals_new_a, satellites_new_a, _ = m_new_a.mock_it()
    t_new_a = time.time() - t0
    print(f"MockerNewPathA took {t_new_a:.4f}s. Centrals: {centrals_new_a.size}, Satellites: {satellites_new_a.size}")

    print("\n--- Running Vectorized MockerNewPathB (Approx Match) ---")
    m_new_b = MockerNewPathB(halos_sample, chunk_size=5000)
    t0 = time.time()
    centrals_new_b, satellites_new_b, _ = m_new_b.mock_it()
    t_new_b = time.time() - t0
    print(f"MockerNewPathB took {t_new_b:.4f}s. Centrals: {centrals_new_b.size}, Satellites: {satellites_new_b.size}")

    print("\n=== Verification on Subsample ===")
    # Check centrals properties
    centrals_match = True
    for prop in ['haloid', 'lgm', 'rvir', 'con', 'x', 'y', 'z', 'Mr', 'Nsat']:
        close = np.allclose(centrals_old[prop], centrals_new_a[prop])
        if not close:
            centrals_match = False
            print(f"Centrals {prop} mismatch!")
    print(f"Centrals match: {centrals_match}")

    # Check satellites properties (Path A)
    satellites_match = True
    for prop in ['haloid', 'lgm', 'con', 'x', 'y', 'z', 'Mr']:
        close = np.allclose(satellites_old[prop], satellites_new_a[prop])
        if not close:
            satellites_match = False
            print(f"Satellites {prop} mismatch!")
    print(f"Satellites match: {satellites_match}")
