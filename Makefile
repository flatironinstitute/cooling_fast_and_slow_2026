# Figure 5, Figure S4 and Table S3: build orchestration.
#
# Two entry points, because the coordinate data lives in the companion Zenodo
# deposit while the derived observables (~8 MB) are tracked in this repository:
#
#   make figures       rebuild every product from the tracked observables.
#                      Needs no coordinates and no download.  This is the
#                      default and what a reader should try first.
#
#   make observables   re-derive those observables from the staged coordinates,
#                      overwriting the tracked copies.  Needs the Zenodo inputs
#                      in place (see directory_structure.txt) plus LOOS.
#
#   make verify        recompute and compare against the tracked observables
#                      WITHOUT overwriting them.  Reports the largest deviation.
#
# Run from anywhere: every script resolves the repository root itself.

PYTHON      ?= python
SEED        ?= 20260403          # for the 230 K per-state sampling

OBS         := derived/observables
FIG5_OBS    := $(OBS)/fig5-observables.npz
S4_OBS      := $(OBS)/fig-s4-trex-populations-k50-t250.npz
S3_OBS      := $(OBS)/table-s3-pbc-self-image.npz

FIG5_OUT    := output/fig5-panels/fig5-4panel-overlay.pdf
S4_OUT      := output/fig-s4-trex-populations/fig-s4-trex-log10-populations-k50-t250.pdf
S3_OUT      := output/table-s3-pbc-self-image/table-s3.tex

COOLED_STAMP := derived/.cooled-by-state.stamp
TREX_STAMP   := derived/.trex-230-by-state.stamp
S277_STAMP   := derived/.sampled-277k-by-state.stamp

.PHONY: all figures observables coordinates verify clean help
.DEFAULT_GOAL := figures

all: figures

help:
	@echo "figures      rebuild Figure 5, Figure S4 and Table S3 from tracked observables"
	@echo "coordinates  rebuild the per-state DCDs (cooled, 230 K, 277 K)"
	@echo "observables  re-derive the observables from coordinates (overwrites tracked npz)"
	@echo "verify       recompute and compare against tracked observables, no overwrite"
	@echo "clean        remove generated figures and tables"

# ── Tier 1: products from the tracked observables ────────────────────────────
figures: $(FIG5_OUT) $(S4_OUT) $(S3_OUT)

$(FIG5_OUT): $(FIG5_OBS) fig5_panels.py paths.py
	$(PYTHON) fig5_panels.py

$(S4_OUT): $(S4_OBS) fig_s4_trex_populations.py paths.py
	$(PYTHON) fig_s4_trex_populations.py

$(S3_OUT): $(S3_OBS) table_s3_pbc_self_image.py fig5_panels.py paths.py
	@mkdir -p $(dir $(S3_OUT))
	$(PYTHON) table_s3_pbc_self_image.py \
	    --md $(dir $(S3_OUT))table-s3.md \
	    --tex $(dir $(S3_OUT))table-s3.tex \
	    --csv $(dir $(S3_OUT))table-s3.csv

# ── Tier 2: rebuild the coordinate-derived intermediates ─────────────────────
coordinates: $(COOLED_STAMP) $(TREX_STAMP) $(S277_STAMP)

$(COOLED_STAMP): build-cooled-by-state.py paths.py
	$(PYTHON) build-cooled-by-state.py
	@touch $@

$(TREX_STAMP): build-trex-230-by-state.py paths.py
	$(PYTHON) build-trex-230-by-state.py --seed $(SEED)
	@touch $@

# 277 K analysed per state.  sampled_frames.xtc cannot be used for table S3:
# it is centred, rotated onto a common reference and carries no box.
$(S277_STAMP): build-277k-by-state.py paths.py
	$(PYTHON) build-277k-by-state.py
	@touch $@

# ── Tier 2: re-derive the observables themselves ─────────────────────────────
observables: coordinates
	$(PYTHON) fig5_panels.py --recompute --stats-only
	$(PYTHON) fig_s4_trex_populations.py --recompute
	$(PYTHON) table_s3_pbc_self_image.py --recompute

verify: coordinates
	$(PYTHON) verify-observables.py

clean:
	rm -rf output/fig5-panels/fig5-* \
	       output/fig-s4-trex-populations/fig-s4-* \
	       output/table-s3-pbc-self-image/table-s3.*
