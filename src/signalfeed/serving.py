"""HN content personalization and hybrid search. No claim of MIND transfer gains."""

import json
import hashlib
import logging
import re
import threading
import time
import numpy as np
from .settings import path, ENCODER_MODEL, ENCODER_REVISION
from .store import db


class ContentService:
    def __init__(self):
        self.lock = threading.RLock()
        self.model = None
        self.items = []
        self.vectors = None
        self.version = "content-v1"
        self.refresh_at = 0
        self.online_ranker = None
        self.online_version = None
        self.embedding_cache = {}
        self.query_cache = {}

    def encode_query(self, text):
        if text not in self.query_cache:
            if len(self.query_cache) >= 128:
                self.query_cache.pop(next(iter(self.query_cache)))
            self.query_cache[text] = self.model.encode(
                [text], normalize_embeddings=True
            )[0]
        return self.query_cache[text]

    def active_ranker(self):
        """Load a verified published bundle, retaining last good model on failure."""
        active = path("online-models") / "active.json"
        if not active.exists():
            self.online_ranker = None
            self.online_version = None
            return None
        try:
            import torch
            from .ranker import Ranker

            manifest = json.loads(active.read_text())
            if self.online_version != manifest["version"]:
                file = path("online-models") / manifest["version"] / "model.pt"
                if (
                    hashlib.sha256(file.read_bytes()).hexdigest()
                    != manifest["model_sha256"]
                ):
                    raise ValueError("Published model checksum does not match manifest")
                candidate = Ranker()
                candidate.load_state_dict(
                    torch.load(file, map_location="cpu", weights_only=True)
                )
                self.online_ranker = candidate.eval()
                self.online_version = manifest["version"]
        except (OSError, ValueError, KeyError, RuntimeError):
            logging.exception("Online bundle reload failed; retaining last good policy")
        return self.online_ranker

    def refresh(self, force=False):
        with self.lock:
            if (
                self.vectors is not None
                and not force
                and time.time() - self.refresh_at < 60
            ):
                return
            from sentence_transformers import SentenceTransformer

            if self.model is None:
                import torch
                from threadpoolctl import threadpool_limits

                torch.set_num_threads(2)
                threadpool_limits(limits=1, user_api="blas")
                local = path("models") / "encoder"
                self.model = SentenceTransformer(
                    str(local) if local.exists() else ENCODER_MODEL,
                    revision=None if local.exists() else ENCODER_REVISION,
                    device="cpu",
                )
            with db() as c:
                self.items = [
                    dict(r)
                    for r in c.execute(
                        "SELECT * FROM articles WHERE deleted=0 ORDER BY id"
                    )
                ]
            missing = [
                a["title"] for a in self.items if a["title"] not in self.embedding_cache
            ]
            if missing:
                encoded = self.model.encode(
                    missing, batch_size=128, normalize_embeddings=True
                )
                self.embedding_cache.update(zip(missing, encoded))
            self.vectors = (
                np.array([self.embedding_cache[a["title"]] for a in self.items])
                if self.items
                else np.empty((0, 384), dtype="float32")
            )
            current_titles = {a["title"] for a in self.items}
            self.embedding_cache = {
                k: v for k, v in self.embedding_cache.items() if k in current_titles
            }
            self.index = {a["id"]: i for i, a in enumerate(self.items)}
            self.refresh_at = time.time()

    def recommend(self, user, query="", limit=30):
        self.refresh()
        with self.lock:
            if not self.items:
                return []
            with db() as c:
                row = c.execute(
                    "SELECT interests FROM users WHERE id=?", (user,)
                ).fetchone()
                interests = json.loads(row[0]) if row else []
                hist = c.execute(
                    "SELECT item_id,MAX(created) stamp FROM events WHERE user_id=? AND kind IN ('click','bookmark') GROUP BY item_id ORDER BY stamp DESC LIMIT 20",
                    (user,),
                ).fetchall()
                bookmarks = {
                    r[0]
                    for r in c.execute(
                        "SELECT item_id FROM bookmarks WHERE user_id=?", (user,)
                    )
                }
                collaborative = {}
                if hist and not query:
                    history_ids = [r[0] for r in hist]
                    placeholders = ",".join("?" for _ in history_ids)
                    collaborative = dict(
                        c.execute(
                            f"SELECT e2.item_id,COUNT(DISTINCT e2.user_id) FROM events e1 "
                            f"JOIN events e2 ON e1.user_id=e2.user_id "
                            f"WHERE e1.item_id IN ({placeholders}) AND e1.user_id<>? "
                            "AND e1.kind IN ('click','bookmark') AND e2.kind IN ('click','bookmark') "
                            "AND e1.is_test=0 AND e2.is_test=0 GROUP BY e2.item_id "
                            "ORDER BY COUNT(DISTINCT e2.user_id) DESC LIMIT 100",
                            (*history_ids, user),
                        ).fetchall()
                    )
                lexical = []
                terms = re.findall(r"\w+", query)[:12]
                if terms:
                    expression = " OR ".join('"' + t + '"' for t in terms)
                    lexical = [
                        r[0]
                        for r in c.execute(
                            "SELECT id FROM article_fts WHERE article_fts MATCH ? ORDER BY bm25(article_fts) LIMIT 100",
                            (expression,),
                        )
                    ]
            hi = [self.index[r[0]] for r in hist if r[0] in self.index]
            profile = self.vectors[hi].mean(0) if hi else np.zeros(384, dtype="float32")
            if interests:
                profile = profile + self.encode_query(" ".join(interests))
            profile = profile / max(np.linalg.norm(profile), 1e-8)
            affinity = self.vectors @ profile
            fresh = np.array(
                [
                    np.exp(-max(0, time.time() - a["published"]) / 86400 / 3)
                    for a in self.items
                ]
            )
            pop = np.log1p([a["score"] for a in self.items])
            pop = pop / max(1, pop.max())
            if query:
                q = self.encode_query(query)
                dense = self.vectors @ q
                scores = np.zeros(len(self.items))
                for rank, i in enumerate(np.argsort(-dense)[:100]):
                    scores[i] += 1 / (61 + rank)
                for rank, iid in enumerate(lexical):
                    if iid in self.index:
                        scores[self.index[iid]] += 1 / (61 + rank)
                order = np.argsort(-(scores + 0.001 * affinity))[:limit]
                reason = "关键词与语义混合检索"
            else:
                scores = 0.6 * affinity + 0.25 * pop + 0.15 * fresh
                if collaborative:
                    scale = max(collaborative.values())
                    for iid, count in collaborative.items():
                        if iid in self.index:
                            scores[self.index[iid]] += 0.15 * count / scale
                # Previously opened items are downweighted but still searchable.
                scores[hi] -= 0.4
                active = self.active_ranker()
                if active is not None:
                    import torch
                    from .online_learning import online_matrix

                    ids = np.argsort(-scores)[:100]
                    with torch.inference_mode():
                        rank_scores = self.online_ranker(
                            torch.from_numpy(online_matrix(self.vectors, ids, hi[::-1]))
                        ).numpy()
                    order = ids[np.argsort(-rank_scores)][:limit]
                    self.version = "online-" + self.online_version
                else:
                    order = np.argsort(-scores)[:limit]
                    self.version = "content-v1"
                    self.online_version = None
                    self.online_ranker = None
                reason = "根据阅读兴趣推荐" if hi or interests else "近期热门技术资讯"
            return [
                {
                    **self.items[i],
                    "bookmarked": self.items[i]["id"] in bookmarks,
                    "reason": reason,
                }
                for i in order
            ]
