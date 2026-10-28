# Third-party provenance

MiniOneRec: https://github.com/AkaliKong/MiniOneRec
Pinned commit: 526306229326013a7ca46f5b2d8f55ea669aa159. Apache-2.0 (LICENSE).
`src/signalfeed/vendor/minionerec/data.py` and `original_logits.py` are copied from this commit. Changes to data.py replace unsafe literal evaluation. The original constrained processor is retained for attribution/comparison; SignalFeed's trie decoder removes model-name prefix offsets and mutable cross-request state.
MiniOneRec credits ReRe and LC-Rec; its original acknowledgement is retained in the cloned upstream repository.

MIND: https://msnews.github.io/ ; Microsoft Research License, research only.
License: https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf
Raw data is external to this repository and must not be redistributed as application content.

Qwen2.5-0.5B and all-MiniLM-L6-v2: Apache-2.0 model cards and licenses retained in downloaded snapshots.
Hacker News API: https://github.com/HackerNews/API . API documentation MIT; linked article content remains with its owners. SignalFeed stores metadata and links, not article bodies.

SignalFeed additions are licensed Apache-2.0. Independent contributions: chronological MIND pipeline, shared feature computation, traditional retrieval, DNN ranking, prefix-to-vector generative retrieval, fusion/ablation evaluation, HN ingest, API, frontend, versioned real feedback and integration tests.
