import argparse
import pandas as pd
from utils import (
    long_to_wide, resample_sync, derive_states, filter_signals,
    finite_diff, trig_and_norm, finalize_export, save_outputs,
    expand_rc_out_to_thrust, add_thrust_forces  # Updated function name
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cutoff", type=float, default=1.2)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--resample-hz", type=float, default=10.0)
    ap.add_argument("--compute-forces", action='store_true',
                    help="Compute thrust forces in Newtons")
    return ap.parse_args()

def run_pipeline(args):
    df_long = pd.read_csv(args.input, sep='\t', engine='python', quoting=3)
    print(f"[INFO] Loaded {len(df_long)} rows")
    print(f"[INFO] Topics: {df_long['topic'].unique()}")
    
    # Check for RC/Out data
    rc_rows = df_long[df_long["topic"].str.contains("rc/out", na=False)]
    print(f"[INFO] Found {len(rc_rows)} RC/Out rows")
    
    # Convert to wide format (includes RC/Out -> thrust conversion)
    wide = long_to_wide(df_long)
    print(f"[INFO] Wide format columns: {list(wide.columns)}")
    
    # Check if thrust data was successfully converted
    thrust_cols = [c for c in wide.columns if 'thrust' in c.lower()]
    print(f"[INFO] Thrust columns: {thrust_cols}")
    
    if not thrust_cols:
        print("[WARNING] No thrust columns found! Check RC/Out conversion.")
        print(f"[WARNING] Available columns: {list(wide.columns)}")
    
    wide, fs = resample_sync(wide, args.resample_hz)
    print(f"[INFO] Resampled to {fs:.2f} Hz")
    
    # Optionally compute forces
    if args.compute_forces:
        if any('cmd_thrust' in c for c in wide.columns):
            wide = add_thrust_forces(wide)
            force_cols = [c for c in wide.columns if 'force' in c or 'moment' in c]
            print(f"[INFO] Added force/moment columns: {force_cols}")
        else:
            print("[WARNING] No thrust commands found, skipping force calculation")
    
    wide = derive_states(wide)
    wide = filter_signals(wide, fs, cutoff=args.cutoff, order=args.order)
    wide = finite_diff(wide, fs)
    wide, stats = trig_and_norm(wide)
    
    df_out = finalize_export(wide)
    save_outputs(df_out, args.output, stats)
    
    print(f"\n✅ SUCCESS: Saved to {args.output}")
    print(f"   Rows: {len(df_out)}")
    print(f"   Sampling rate: {fs:.2f} Hz")
    print(f"   Columns: {len(df_out.columns)}")
    
    # Show thrust statistics if available
    for col in ['cmd_thrust.port', 'cmd_thrust.stbd']:
        if col in df_out.columns:
            print(f"   {col}: min={df_out[col].min():.2f}, max={df_out[col].max():.2f}, mean={df_out[col].mean():.2f}")

def main():
    args = parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()