"""HN content personalization and hybrid search. No claim of MIND transfer gains."""

import json
import hashlib
import io
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
        self.refresh_lock = threading.Lock()
        self.encode_lock = threading.Lock()
        self.load_error = None
        self.refresh_error = None
        self.bundle_key = None
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
        with self.encode_lock:
            if text not in self.query_cache:
                if len(self.query_cache) >= 128:
                    self.query_cache.pop(next(iter(self.query_cache)))
                self.query_cache[text] = self.model.encode(
                    [text], normalize_embeddings=True
                )[0]
            return self.query_cache[text]

    def active_ranker(self):
        with self.lock:
            return self._active_ranker()

    def _active_ranker(self, contract=None, directory="online-models"):
        active = path(directory) / "active.json"
        if not active.exists():
            self.online_ranker = self.online_version = self.bundle_key = None
            self.load_error = None
            return None
        try:
            import torch
            from .ranker import Ranker
            from .online_schema import CONTRACT

            expected = CONTRACT if contract is None else contract

            manifest = json.loads(active.read_text())
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise ValueError(
                    "Incompatible online model/feature/encoder/policy contract"
                )
            version = manifest["version"]
            if not isinstance(version, str) or not re.fullmatch(
                r"[A-Za-z0-9_-]+", version
            ):
                raise ValueError("Invalid bundle version")
            key = (version, manifest["model_sha256"])
            if self.bundle_key != key:
                payload = (path(directory) / version / "model.pt").read_bytes()
                if hashlib.sha256(payload).hexdigest() != manifest["model_sha256"]:
                    raise ValueError("Published model checksum mismatch")
                candidate = Ranker()
                candidate.load_state_dict(
                    torch.load(
                        io.BytesIO(payload), map_location="cpu", weights_only=True
                    )
                )
                candidate.eval()
                if not all(torch.isfinite(p).all() for p in candidate.parameters()):
                    raise ValueError("Non-finite model weights")
                with torch.inference_mode():
                    probe = candidate(
                        torch.stack([torch.zeros(1158), torch.ones(1158)])
                    )
                if probe.shape != (2,) or not torch.isfinite(probe).all():
                    raise ValueError("Invalid model smoke prediction")
                self.online_ranker, self.online_version, self.bundle_key = (
                    candidate,
                    version,
                    key,
                )
            self.load_error = None
        except Exception as exc:
            # Deserialization also raises EOFError/UnpicklingError/TypeError.
            self.load_error = type(exc).__name__ + ": " + str(exc)
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
            cold = self.vectors is None
        if not self.refresh_lock.acquire(blocking=cold):
            return
        try:
            from sentence_transformers import SentenceTransformer

            with self.encode_lock:
                if self.model is None:
                    import torch
                    from threadpoolctl import threadpool_limits

                    torch.set_num_threads(2)
                    threadpool_limits(limits=1, user_api="blas")
                    # Always bind the actual encoder to the published revision.
                    try:
                        self.model = SentenceTransformer(
                            ENCODER_MODEL,
                            revision=ENCODER_REVISION,
                            device="cpu",
                            local_files_only=True,
                        )
                    except OSError:
                        self.model = SentenceTransformer(
                            ENCODER_MODEL, revision=ENCODER_REVISION, device="cpu"
                        )
            with db() as c:
                items = [
                    dict(r)
                    for r in c.execute(
                        "SELECT * FROM articles WHERE deleted=0 ORDER BY id"
                    )
                ]
            cache = dict(self.embedding_cache)
            missing = list(
                dict.fromkeys(a["title"] for a in items if a["title"] not in cache)
            )
            if missing:
                with self.encode_lock:
                    encoded = self.model.encode(
                        missing, batch_size=128, normalize_embeddings=True
                    )
                if (
                    np.asarray(encoded).shape != (len(missing), 384)
                    or not np.isfinite(encoded).all()
                ):
                    raise ValueError("Invalid encoder output")
                cache.update(zip(missing, encoded))
            vectors = (
                np.array([cache[a["title"]] for a in items], dtype="float32")
                if items
                else np.empty((0, 384), dtype="float32")
            )
            index = {a["id"]: i for i, a in enumerate(items)}
            if (
                len(index) != len(items)
                or vectors.shape != (len(items), 384)
                or not np.isfinite(vectors).all()
            ):
                raise ValueError("Inconsistent catalog snapshot")
            vectors.setflags(write=False)
            with self.lock:
                self.items, self.vectors, self.index = items, vectors, index
                self.embedding_cache = {a["title"]: cache[a["title"]] for a in items}
                self.refresh_at, self.refresh_error = time.time(), None
        except Exception as exc:
            self.refresh_error = type(exc).__name__ + ": " + str(exc)
            if self.vectors is None:
                raise
            logging.exception(
                "Catalog refresh failed; complete previous snapshot retained"
            )
        finally:
            self.refresh_lock.release()

    def recommend(self, user, query="", limit=30, *, snapshot=None):
        self.refresh()
        with self.lock:
            items, vectors, index = self.items, self.vectors, self.index
        if not items:
            return []
        created = time.time()
        with db() as c:
            c.execute("BEGIN")
            row = c.execute(
                "SELECT interests FROM users WHERE id=?", (user,)
            ).fetchone()
            interests = json.loads(row[0]) if row else []
            hist = c.execute(
                "SELECT item_id,MAX(created) stamp FROM events WHERE user_id=? AND created<=? AND kind IN ('click','bookmark') GROUP BY item_id ORDER BY stamp DESC,item_id LIMIT 20",
                (user, created),
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
                        "AND e1.is_test=0 AND e2.is_test=0 AND e1.created<=? AND e2.created<=? GROUP BY e2.item_id "
                        "ORDER BY COUNT(DISTINCT e2.user_id) DESC LIMIT 100",
                        (*history_ids, user, created, created),
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
        hi = [index[r[0]] for r in hist if r[0] in index]
        profile = vectors[hi].mean(0) if hi else np.zeros(384, dtype="float32")
        explicit = (
            self.encode_query(" ".join(interests))
            if interests
            else np.zeros(384, dtype="float32")
        )
        profile = profile + explicit
        profile = profile / max(np.linalg.norm(profile), 1e-8)
        affinity = vectors @ profile
        fresh = np.array(
            [np.exp(-max(0, created - a["published"]) / 86400 / 3) for a in items]
        )
        pop = np.log1p([a["score"] for a in items])
        pop = pop / max(1, pop.max())
        if query:
            q = self.encode_query(query)
            dense = vectors @ q
            scores = np.zeros(len(items))
            for rank, i in enumerate(np.argsort(-dense)[:100]):
                scores[i] += 1 / (61 + rank)
            for rank, iid in enumerate(lexical):
                if iid in index:
                    scores[index[iid]] += 1 / (61 + rank)
            order = np.argsort(-(scores + 0.001 * affinity))[:limit]
            reason = "关键词与语义混合检索"
        else:
            scores = 0.6 * affinity + 0.25 * pop + 0.15 * fresh
            if collaborative:
                scale = max(collaborative.values())
                for iid, count in collaborative.items():
                    if iid in index:
                        scores[index[iid]] += 0.15 * count / scale
            # Previously opened items are downweighted but still searchable.
            scores[hi] -= 0.4
            with self.lock:
                active = self.active_ranker()
                version = (
                    "online-" + self.online_version
                    if active is not None
                    else "content-v1"
                )
            if active is not None:
                from .online_schema import online_matrix, policy_scores

                ids = np.argsort(-scores, kind="stable")[:100]
                ranking_features = online_matrix(
                    vectors, ids, hi[::-1], explicit, scores[ids]
                )
                rank_scores = policy_scores(active, ranking_features, scores[ids])
                order = ids[np.argsort(-rank_scores, kind="stable")][:limit]
                feature_rows = dict(zip(ids, ranking_features))
                self.version = version
            else:
                order = np.argsort(-scores, kind="stable")[:limit]
                self.version = "content-v1"
            reason = "根据阅读兴趣推荐" if hi or interests else "近期热门技术资讯"
            if snapshot is not None:
                from .online_schema import FEATURE_VERSION, online_matrix

                saved_features = (
                    np.stack([feature_rows[i] for i in order])
                    if active is not None
                    else online_matrix(
                        vectors, order, hi[::-1], explicit, scores[order]
                    )
                )
                snapshot.update(
                    created=created,
                    feature_version=FEATURE_VERSION,
                    model_version=version,
                    history=[items[i]["id"] for i in hi[::-1]],
                    interests=interests,
                    # Scores include interests, popularity, freshness, CF
                    # and read penalties, frozen before feedback arrives.
                    content_scores={items[i]["id"]: float(scores[i]) for i in order},
                    features=dict(
                        zip(
                            [items[i]["id"] for i in order],
                            saved_features.tolist(),
                        )
                    ),
                )
        return [
            {
                **items[i],
                "bookmarked": items[i]["id"] in bookmarks,
                "reason": reason,
            }
            for i in order
        ]
