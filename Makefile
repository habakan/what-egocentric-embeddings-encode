# Reproduce the paper. Each classifier is a file target, so finished steps are skipped.
#   make classifiers   preprocessing and all per-clip classifiers (out-of-fold predictions)
#   make table3        main results (Table 3) and the data for Figs. 5 and 7
#   make figures       all figures
#   make sensitivity   decision-rule prior, test-aligned video rows and whitening (Section 7.5)
#   make submission    the paper's configuration as a submission file
#   make competition   the final challenge submission (paper system + count constraint between stages)
# Training commands were recovered from the development runs; rotation angles and seeds differ by run.

PY  := uv run python
EXP := experiments
PREP := $(EXP)/prep

INERTIAL := exp001_lgb_inertial exp014_nn_inertial_rot exp015_nn_inertial_s1337 exp016_nn_inertial_rot30 \
            exp017_nn_aux exp018_nn_aux_s99 exp019_nn_aux_s555
VIDEO    := exp013_nn_video_5fold exp020_video_only exp022_vonly_center_5f
EXTRA    := teacher01 exp021_distill_w05

.PHONY: all prep classifiers extra table3 figures sensitivity submission competition clean-results
all: table3 figures

# ---------- preprocessing
prep: $(PREP)/win_meta.parquet $(PREP)/video_raw_head.npy $(PREP)/video_raw_center.npy $(PREP)/video_raw_test.npy

$(PREP)/win_meta.parquet:
	$(PY) src/prepare/prepare_windows.py
	$(PY) src/prepare/prepare_valid_mask.py
	$(PY) src/prepare/prepare_video.py
	$(PY) src/prepare/prepare_video.py --test

$(PREP)/video_raw_head.npy: $(PREP)/win_meta.parquet
	$(PY) src/prepare/prepare_video_raw.py --offset head

$(PREP)/video_raw_center.npy: $(PREP)/win_meta.parquet
	$(PY) src/prepare/prepare_video_raw.py --offset center

$(PREP)/video_raw_test.npy: $(PREP)/win_meta.parquet
	$(PY) src/prepare/prepare_video_raw.py --test

# ---------- per-clip classifiers (out-of-fold)
# the aux networks read the hand-crafted features that train_lgb.py caches in prep/feat_inertial.npy
classifiers: $(foreach t,$(INERTIAL) $(VIDEO),$(EXP)/$(t)/oof.npy)
extra: $(EXP)/teacher01/oof_win.npy $(EXP)/exp021_distill_w05/oof.npy

$(EXP)/exp001_lgb_inertial/oof.npy: | prep
	$(PY) src/train/train_lgb.py --tag exp001_lgb_inertial

$(EXP)/exp014_nn_inertial_rot/oof.npy: | prep
	$(PY) src/train/train_nn.py --folds 5 --epochs 25 --rot 20 --tag exp014_nn_inertial_rot
$(EXP)/exp015_nn_inertial_s1337/oof.npy: | prep
	$(PY) src/train/train_nn.py --folds 5 --epochs 25 --rot 20 --seed 1337 --tag exp015_nn_inertial_s1337
$(EXP)/exp016_nn_inertial_rot30/oof.npy: | prep
	$(PY) src/train/train_nn.py --folds 5 --epochs 30 --rot 30 --seed 2024 --tag exp016_nn_inertial_rot30

$(EXP)/exp017_nn_aux/oof.npy: $(EXP)/exp001_lgb_inertial/oof.npy | prep
	$(PY) src/train/train_nn.py --folds 5 --epochs 25 --rot 20 --aux --seed 7 --tag exp017_nn_aux
$(EXP)/exp018_nn_aux_s99/oof.npy: $(EXP)/exp001_lgb_inertial/oof.npy | prep
	$(PY) src/train/train_nn.py --folds 5 --epochs 25 --rot 20 --aux --seed 99 --tag exp018_nn_aux_s99
$(EXP)/exp019_nn_aux_s555/oof.npy: $(EXP)/exp001_lgb_inertial/oof.npy | prep
	$(PY) src/train/train_nn.py --folds 5 --epochs 30 --rot 25 --aux --seed 555 --tag exp019_nn_aux_s555

# both modalities (probabilities U enter the graph features)
$(EXP)/exp013_nn_video_5fold/oof.npy: | prep
	$(PY) src/train/train_nn.py --folds 5 --epochs 12 --video --tag exp013_nn_video_5fold
# video only, two clip offsets (probabilities V are pooled after propagation)
$(EXP)/exp020_video_only/oof.npy: | prep
	$(PY) src/train/train_nn.py --folds 5 --epochs 14 --video --no-inertial --seed 11 --tag exp020_video_only
$(EXP)/exp022_vonly_center_5f/oof.npy: | prep
	$(PY) src/train/train_nn.py --folds 5 --epochs 12 --video --no-inertial --offset center --tag exp022_vonly_center_5f

# four-sensor teacher and distillation (Section 7.3, Appendix C)
$(EXP)/teacher01/oof_win.npy: | prep
	$(PY) src/train/train_teacher.py --tag teacher01
$(EXP)/exp021_distill_w05/oof.npy: $(EXP)/teacher01/oof_win.npy
	$(PY) src/train/train_nn.py --folds 5 --epochs 25 --rot 20 --aux --seed 7 \
	  --distill teacher01 --distill-w 0.5 --distill-t 3.0 --tag exp021_distill_w05

# ---------- results
table3: paper/figdata/prf.json
paper/figdata/prf.json: classifiers
	$(PY) src/sec7_results/probe_prf.py

figures: table3
	$(PY) paper/make_figures.py

# decision-rule prior per participant, test-aligned video rows, whitening (Section 7.5, App. D item 9)
sensitivity: paper/figdata/review_sensitivity.json
paper/figdata/review_sensitivity.json: classifiers $(PREP)/video_raw_sync.npy
	$(PY) src/sec7_results/repro_review_sensitivity.py

$(PREP)/video_raw_sync.npy: $(PREP)/win_meta.parquet
	$(PY) src/prepare/prepare_video_raw.py --offset sync

submission: classifiers
	$(PY) src/submit/final_submit.py --tags $(INERTIAL) --weights 1.5 1 1 1 1 1 1 \
	  --vgraph exp013_nn_video_5fold --temp 1.4 --vblend exp020_video_only exp022_vonly_center_5f \
	  --vblend-where every --g0 0.2 --graph-drop-pcs 30 --out exp015_graphpc30

# final challenge submission: the count constraint applied between the propagation stages,
# lambda = 2 and the share band of the exercise-variant pairs (public leaderboard 0.894)
competition: classifiers
	$(PY) src/submit/submit_band_inloop.py --lam 2 --pair --out exp028_band_inloop_pair

clean-results:
	rm -f paper/figdata/prf.json paper/figdata/preds.npz
