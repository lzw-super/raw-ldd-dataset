# SID Sony synthetic-noise training

The training route follows the CVPR 2025 paper's clean-long-RAW plus
real-dark-frame setup. It deliberately excludes SID short RAW from the loss.
Clean patches are sampled dynamically from the 161 full-resolution train long
RAW files. The 20 prefix-`2` long RAW scenes form an independent synthetic
validation set.

Use the existing environment:

```bash
conda run --no-capture-output -n LED-ICCV23 python tools/build_sid_train_index.py
conda run --no-capture-output -n LED-ICCV23 python tools/validate_sid_training.py
```

The index must report 161 prefix-`0` training scenes and 20 prefix-`2`
validation scenes. Eight new crops per training scene give 1,288 steps/epoch.
All training ISO values should map exactly to an available LLD dark-frame ISO.

Run a one-epoch smoke training first:

```bash
conda run --no-capture-output -n LED-ICCV23 python train_sid_sony.py \
  --config configs/train_sid_sony.yaml \
  --output-dir experiments/smoke_sid_sony \
  --epochs 1 --steps-per-epoch 8 --num-workers 0 \
  --validate-every 1 --validate-steps 2 --save-every 1
```

The formal ratio-aware run (start from scratch) is:

```bash
conda run --no-capture-output -n LED-ICCV23 python train_sid_sony.py \
  --config configs/train_sid_sony.yaml --rebuild-manifest
```

It has 1,288 steps/epoch (161 clean scenes × 8 dynamic full-RAW crops).
`latest.pth` is
updated after every epoch; only the three most recent numbered checkpoints are
kept, which bounds disk use.  Resume with `--resume <run>/checkpoints/latest.pth`.
Resume only with the same epoch/step schedule; start a new run for the formal
1,000-epoch configuration after a deliberately shortened smoke run.

Do not resume a checkpoint produced by the old fixed-NPZ pipeline. If it is
useful as an initialization for a diagnostic fine-tune, pass it with
`--init-checkpoint` and use a new output directory; optimizer/scheduler state
must not be imported.

For the paper-literal ratio ablation, change `--synthesis paper_literal` and
give it a different `--output-dir`.  Do not mix this with the main checkpoint.

Validate a trained model against official real SID pairs in all three groups:

```bash
for ratio in 100 250 300; do
  conda run --no-capture-output -n LED-ICCV23 python test_denoise_sideld.py \
    --cp-dir experiments/sid_sony_paper_fair/checkpoints/latest.pth \
    --testset-type sid \
    --eval-ratio "$ratio" \
    --device cuda:1 \
    --num-workers 2 \
    --result-json "experiments/sid_sony_paper_fair/sid_x${ratio}.json"
done
```

The test script uses `infos/SID_evaltest.info`, PMN dark shading, the official
RAW preprocessing order, and ELD global illuminance correction.  It is a real
SID metric rather than the small synthetic sanity metric logged during training.
