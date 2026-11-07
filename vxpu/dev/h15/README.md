# Hero 15, pod C's last chains (Nov 6 to 7, 2026)

Pod C (hero7d) was evicted at about 00:21 UTC on Nov 7 for ephemeral storage, most likely a core dump from
F105's crash written into the container layer, and everything on it was lost. These files let the morning rebuild
the last runs from the branch.

- `chainC180.sh`, `watchC180.sh`: F106a (slots by survival from the acceptance estimator, no cost model, D94's
  path) then F106 (the cost model with the plan constants, the program-choice check and rank 0's rows for every
  rank). They expect the hero 15 pod layout (`/root/hero/h15/tpcommon3.sh`, `f97.env`, `table-sweep16.json`, the
  flats `spec3f112-tp4-b16` and `spec3f128-tp4-b16`, the fork at `/mnt/ssd/deepseek_v41-h15spec7/inference`) and the
  drafter export `spec3t8-est-tp4-b16`, made with `VXPU_DRAFT_CONF=maxprob` and
  `--step-tokens 8 --draft-sharded --draft-markov sharded --draft-head fp8` on the common export arguments.
- `plans.json`: the plan files the chains name (write each key as its own file under `/root/hero/h15/plans`).
- `est_patch.py`, `check_patch.py`, `bcast_patch.py`, `scope_patch.py`: the four edits as applied to the pod's tree
  copies, in order; the branch already carries them, so a tree built from the branch needs none of them.

The pod's entrypoint should set `ulimit -c 0`: a core dump of a 90 GB process fills the node's ephemeral storage.
