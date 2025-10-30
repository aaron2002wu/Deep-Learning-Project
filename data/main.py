import argparse
import pandas as pd
from utils import (
    long_to_wide, resample_sync, derive_states, filter_signals,
    finite_diff, trig_and_norm, finalize_export, save_outputs,expand_cmd_thrust
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cutoff", type=float, default=1.2)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--resample-hz", type=float, default=10.0)
    return ap.parse_args()

def run_pipeline(args):
    df_long = pd.read_csv(args.input, sep='\t', engine='python', quoting=3)
    print(df_long.columns)
    print(df_long[df_long["topic"].isna()])
    thrust_expanded = expand_cmd_thrust(df_long)
    if not thrust_expanded.empty:
        print(thrust_expanded.head(5))   
        df_long = pd.concat([df_long, thrust_expanded], ignore_index=True)

    wide = long_to_wide(df_long)
    print(wide.columns)

    wide, fs = resample_sync(wide, args.resample_hz)

    wide = derive_states(wide)
    wide = filter_signals(wide, fs, cutoff=args.cutoff, order=args.order)
    wide = finite_diff(wide, fs)
    wide, stats = trig_and_norm(wide)

    df_out = finalize_export(wide)
    save_outputs(df_out, args.output, stats)

    print(f"Saved to {args.output} (N={len(df_out)}), fs≈{fs:.2f} Hz")

def main():
    args = parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
