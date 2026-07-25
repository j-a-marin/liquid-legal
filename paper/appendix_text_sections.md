# Text sections for exit-paper completion (draft for parent merge)

## Ethical and deployment-limitations section

This work is a methods and negative-results study on fully synthetic data.
No real docket, party, judge, or litigation outcome was used, and no claim
in this paper transfers to real litigation. Three deployment-relevant
concerns nevertheless deserve explicit statement, because the motivating
application — litigation finance — is one where model errors are
consequential and asymmetrically distributed.

1. **Synthetic-to-real gap.** The generators encode stated, stylized
   mechanisms (judge backlog episodes, adverse case regimes, discovery
   stalls). Real dockets add selection effects, strategic behavior, sealing,
   missing and duplicated entries, and outcome-recording biases that no
   current component of this program measures. Any real-data successor must
   re-derive outcome and censoring definitions under audit before any
   predictive claim is made, per the real-docket bridge requirements.
2. **Fairness and institutional encoding.** Models trained on docket
   histories can encode jurisdiction-, judge-, and party-level proxies for
   protected or legally irrelevant characteristics. The counterfactual judge
   probe demonstrated here is a diagnostic of exactly such encoding; in a
   deployment context it would be an audit obligation, not a feature. The
   program's explanation-correctness checks are a minimum bar, not a
   sufficient one, for regulated use.
3. **Decision-context calibration.** A settlement-probability estimate is
   used to price capital. Miscalibration at decision-relevant horizons has
   direct financial and access-to-justice externalities. This is why the
   evaluation protocol treats calibration (ECE) and duration error as kill
   guards rather than secondary curiosities, and why any successor must
   report calibration by horizon and institution.

Nothing in this repository is legal, financial, or investment advice, and
the killed candidate must not be deployed under any label.

## Artifact availability and licensing statement

All artifacts needed to inspect or recompute every number in this paper are
in the repository:

- Source code: `src/liquid_legal/` (MIT license, see `LICENSE`; `NOTICE`
  covers third-party attributions).
- Program record: `experiments/RESULTS.md`, `PREREGISTRATION.md`,
  `FREEZE.md`, `STAGE1_SPEC.md`, `STAGE1_RESULTS.md`, with SHA-256 hashes of
  the frozen files in `experiments/results/freeze_hashes.txt`.
- Killed-candidate archive: `experiments/archive/stage1-killed/` — 40 weight
  files, 40 prediction files, run logs, code snapshots, deterministic
  reproduction check, and `hashes.json` over the full archive (verified
  91/91 on successor intake, 2026-07-25).
- Exploratory forensics: `experiments/archive/f1-forensics/` with its own
  manifest.
- Figures regenerate via `paper/figures/make_figures.py`.

Verified installation (clean environment): `pip install -e
".[dev,experiments]"`, then `pytest`. Synthetic generators produce all data;
no external dataset is required. There is no proprietary component.

## Author contributions and acknowledgments

[Author names and affiliations to be inserted at submission. CRediT roles:
conceptualization, methodology, software, validation, formal analysis,
investigation, data curation, writing — original draft, writing — review &
editing. The program was designed, executed, and archived by the project
team; the successor team performed archive verification, the independent
calculation of the primary statistic, the exploratory F1 forensics, and the
preparation of this manuscript for publication.]

The authors thank the contributors to the `ncps` library; a
batched-timespans fix developed during this program was contributed back
upstream (staged PR in `.ncps-upstream/`). No external funding supported
this work [confirm at submission]. The authors declare no competing
interests [confirm at submission].
