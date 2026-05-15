#!/usr/bin/env bash
# Sweep all HUGSIM nuscenes scenarios and record the model's frame-0 implied
# velocity (wp0 distance / 0.5s, signed by wp0_dx). For each scenario, the AD
# subprocess (compute_implied_velocity_hugsim.py) runs the agent once, appends
# a CSV row, and sends None on plan_pipe so closed_loop.py exits cleanly.
#
# Run from the HUGSIM dir in the HUGSIM pixi env:
#   pixi run bash compute_implied_velocity_hugsim.sh [CHECKPOINT] [OUTPUT_BASE] [N_SCENARIOS]
#
# DATASET defaults to nuscenes.

module load cuda/11.8/cudnn/8.9
export PYTHONNOUSERSITE=1

DATASET="${DATASET:-nuscenes}"
sim_cuda=0
ad_cuda=0
scenario_dir=/network/scratch/g/grandhia/hugsim_data_old/scenarios/${DATASET}/

ad_checkpoint_path="${1:-/network/scratch/l/luke.rowe/experiments/config_may13_3/checkpoints/last_drivor_compatible.ckpt}"
output_base="${2:-/network/scratch/l/luke.rowe/hugsim_output/implied_velocity/config_may13_3_${DATASET}}"
n_scenarios="${3:-100}"   # default = all

csv_path="${output_base}/implied_velocity.csv"
mkdir -p "${output_base}"
rm -f "${csv_path}"
export DRIVOR_IMPLIED_VELO_CSV="${csv_path}"

echo "DATASET=${DATASET}"
echo "ad_checkpoint_path=${ad_checkpoint_path}"
echo "output_base=${output_base}"
echo "n_scenarios=${n_scenarios}"
echo "csv=${csv_path}"

count=0
for cfg in ${scenario_dir}/*.yaml; do
    if [ "${count}" -ge "${n_scenarios}" ]; then break; fi
    count=$((count + 1))
    echo "[${count}/${n_scenarios}] ${cfg}"
    CUDA_VISIBLE_DEVICES=${sim_cuda} \
    python3 closed_loop.py --scenario_path ${cfg} \
                        --base_path ./configs/sim/${DATASET}_base_implied_velocity.yaml \
                        --camera_path ./configs/sim/${DATASET}_camera.yaml \
                        --kinematic_path ./configs/sim/kinematic.yaml \
                        --ad ltf \
                        --ad_cuda ${ad_cuda} \
                        --ad_checkpoint_path ${ad_checkpoint_path} \
                        --output_dir ${output_base} \
                        --image_size 434 252 \
        || echo "  scenario ${cfg} failed; continuing"
done

echo "Done. CSV at ${csv_path}"
echo "Aggregate stats:"
python3 - "${csv_path}" <<'PY'
import csv, math, sys
path = sys.argv[1]
signed, mag, vel_x, n_reverse, n_total = [], [], [], 0, 0
with open(path) as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            s = float(row["implied_velo_signed"]); m = float(row["implied_velo_mag"]); v = float(row["vel_x"])
        except Exception:
            continue
        if math.isnan(s):
            continue
        signed.append(s); mag.append(m); vel_x.append(v)
        n_total += 1
        if s < 0:
            n_reverse += 1

def stats(name, xs):
    if not xs:
        print(f"  {name}: empty"); return
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    mean = sum(xs_sorted) / n
    var = sum((x - mean) ** 2 for x in xs_sorted) / max(n - 1, 1)
    std = math.sqrt(var)
    print(f"  {name}: n={n} mean={mean:.3f} std={std:.3f} median={xs_sorted[n//2]:.3f} "
          f"min={xs_sorted[0]:.3f} max={xs_sorted[-1]:.3f}  "
          f"p10={xs_sorted[max(0,n//10)]:.3f} p90={xs_sorted[min(n-1,(9*n)//10)]:.3f}")

print(f"  n_scenarios={n_total}  n_reverse={n_reverse} ({100.0*n_reverse/max(n_total,1):.1f}%)")
stats("implied_velo_signed (m/s)", signed)
stats("implied_velo_mag    (m/s)", mag)
stats("ego vel_x           (m/s)", vel_x)
PY
