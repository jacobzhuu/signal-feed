import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { eventId } from "./event-id";
import "./style.css";
type Article = {
  id: string;
  title: string;
  url: string;
  domain: string;
  published: number;
  score: number;
  bookmarked: boolean;
  position: number;
  reason: string;
};
type Listing = { request_id: string; items: Article[]; experiment?: {status: string} };
async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });
  if (!response.ok) throw Error(`请求失败 (${response.status})`);
  return response.json();
}
function Card({
  item,
  request,
  onBookmark,
}: {
  item: Article;
  request: string;
  onBookmark: (id: string, saved: boolean) => void;
}) {
  const ref = useRef<HTMLElement>(null);
  const emitted = useRef(false);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const emit = async (kind: string) => {
    const body = JSON.stringify({
      event_id: eventId(),
      request_id: request,
      item_id: item.id,
      kind,
      position: item.position,
    });
    // Reuse the operation UUID if the response was lost after a successful commit.
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        return await api("/api/events", {
          method: "POST",
          body,
          keepalive: true,
        });
      } catch (error) {
        if (attempt === 1) throw error;
        await new Promise((resolve) => setTimeout(resolve, 200));
      }
    }
  };
  useEffect(() => {
    emitted.current = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let intersecting = false;
    const schedule = () => {
      clearTimeout(timer);
      if (
        intersecting &&
        document.visibilityState === "visible" &&
        !emitted.current
      ) {
        timer = setTimeout(() => {
          if (document.visibilityState === "visible")
            emit("impression")
              .then(() => {
                emitted.current = true;
              })
              .catch(() => {});
        }, 1000);
      }
    };
    const observer = new IntersectionObserver(
      (entries) => {
        intersecting =
          entries[0].isIntersecting && entries[0].intersectionRatio >= 0.5;
        schedule();
      },
      { threshold: [0, 0.5] },
    );
    document.addEventListener("visibilitychange", schedule);
    if (ref.current) observer.observe(ref.current);
    return () => {
      clearTimeout(timer);
      observer.disconnect();
      document.removeEventListener("visibilitychange", schedule);
    };
  }, [request, item.id]);
  return (
    <article ref={ref} className="card">
      <div className="card-top">
        <span>{item.domain.replace(/^www\./, "")}</span>
        <span>{new Date(item.published * 1000).toLocaleDateString()}</span>
      </div>
      <a
        className="story"
        href={item.url}
        target="_blank"
        rel="noreferrer"
        onClick={() => {
          emit("click").catch(() => setError("点击记录失败，请稍后重试"));
        }}
      >
        {item.title}
        <span className="arrow">↗</span>
      </a>
      <div className="card-bottom">
        <span>
          {item.score} HN points <span className="dot">·</span> {item.reason}
        </span>
        <button
          disabled={saving}
          aria-label={item.bookmarked ? "取消收藏" : "收藏文章"}
          className={item.bookmarked ? "saved" : ""}
          onClick={async () => {
            setSaving(true);
            try {
              await emit(item.bookmarked ? "unbookmark" : "bookmark");
              onBookmark(item.id, !item.bookmarked);
            } catch (e) {
              setError(String(e));
            } finally {
              setSaving(false);
            }
          }}
        >
          {item.bookmarked ? "★ 已收藏" : "☆ 收藏"}
        </button>
      </div>
      {error && <p role="alert">{error}</p>}
    </article>
  );
}
function App() {
  const [listing, setListing] = useState<Listing>({
    request_id: "",
    items: [],
  });
  const [q, setQ] = useState("");
  const [tab, setTab] = useState("feed");
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [topics, setTopics] = useState<string[]>([]);
  const topicsRef = useRef<string[]>([]);
  const interestRevision = useRef(0);
  const interestQueue = useRef<Promise<void>>(Promise.resolve());
  const listingSequence = useRef(0);
  const listingUrl = useRef("/api/feed");
  const [stats, setStats] = useState<{
    impressions: number;
    ctr: number | null;
  }>();
  const load = async (url = "/api/feed") => {
    const sequence = ++listingSequence.current;
    listingUrl.current = url;
    setBusy(true);
    setError("");
    try {
      const next = await api<Listing>(url);
      if (sequence === listingSequence.current) setListing(next);
    } catch (e) {
      if (sequence === listingSequence.current) setError(String(e));
    } finally {
      if (sequence === listingSequence.current) setBusy(false);
    }
  };
  useEffect(() => {
    load();
    api<{ topics: string[] }>("/api/interests")
      .then((r) => {
        if (interestRevision.current === 0) {
          topicsRef.current = r.topics;
          setTopics(r.topics);
        }
      })
      .catch(() => {});
    api<typeof stats>("/api/stats")
      .then(setStats)
      .catch(() => {});
    return () => {
      listingSequence.current++;
    };
  }, []);
  const choose = (topic: string) => {
    const sequence = listingSequence.current;
    const revision = ++interestRevision.current;
    const next = topicsRef.current.includes(topic)
      ? topicsRef.current.filter((t) => t !== topic)
      : [...topicsRef.current, topic];
    topicsRef.current = next;
    setTopics(next);
    // Full-list writes must reach the server in the order of user actions.
    interestQueue.current = interestQueue.current.then(async () => {
      try {
        await api("/api/interests", {
          method: "PUT",
          body: JSON.stringify({ topics: next }),
        });
        if (
          revision === interestRevision.current &&
          sequence === listingSequence.current
        ) {
          const cold = listingUrl.current === "/api/feed?cold=true";
          setTab(cold ? "cold" : "feed");
          await load(cold ? "/api/feed?cold=true" : "/api/feed");
        }
      } catch (e) {
        if (revision === interestRevision.current) {
          try {
            const saved = await api<{ topics: string[] }>("/api/interests");
            if (revision === interestRevision.current) {
              topicsRef.current = saved.topics;
              setTopics(saved.topics);
            }
          } catch {
            /* Preserve visible intent, and report the failed save. */
          }
          if (sequence === listingSequence.current) setError(String(e));
        }
      }
    });
  };
  return (
    <>
      <header>
        <a className="brand" href="/">
          <span className="brand-icon">s</span>SignalFeed
          <span className="badge">LIVE</span>
        </a>
        <nav>
          <button
            className={tab === "feed" ? "active" : ""}
            onClick={() => {
              setTab("feed");
              setQ("");
              load();
            }}
          >
            发现
          </button>
          <button
            className={tab === "cold" ? "active" : ""}
            onClick={() => {
              setTab("cold");
              setQ("");
              load("/api/feed?cold=true");
            }}
          >
            兴趣加速（实验）
          </button>
          <button
            className={tab === "bookmarks" ? "active" : ""}
            onClick={() => {
              setTab("bookmarks");
              load("/api/bookmarks");
            }}
          >
            我的收藏
          </button>
        </nav>
        <div className="identity">
          <span />
          你的专属资讯流
        </div>
      </header>
      <main>
        <section className="intro">
          <div className="eyebrow">LESS NOISE. MORE SIGNAL.</div>
          <h1>
            找到值得关注的<span>技术信号。</span>
          </h1>
          <p>真实社区资讯，随你的兴趣生长。阅读、探索，把好内容留下。</p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              setTab("search");
              load("/api/search?q=" + encodeURIComponent(q));
            }}
          >
            <span>⌕</span>
            <input
              aria-label="搜索资讯"
              placeholder="搜索技术、项目或观点，例如 AI agents"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
            <button type="submit">搜索 ↗</button>
          </form>
        </section>
        <div className="layout">
          <section>
            <div className="section-heading">
              <h2>
                {tab === "bookmarks"
                  ? "你的收藏"
                  : tab === "search"
                    ? `搜索：${q}`
                    : tab === "cold" ? "兴趣加速"
                    : "为你发现"}
              </h2>
              <span>
                {busy ? "正在加载…" : `${listing.items.length} 篇资讯`}
              </span>
            </div>
            {tab === "cold" && <div className="panel experiment-note" role="status">
              <p>从兴趣和最近阅读开始，逐步调整推荐。这是实验功能，尚未验证实际效果。</p>
              {listing.experiment?.status === "profile-warming-fallback"
                ? <p>兴趣画像正在准备，先展示基础推荐。稍后可更新。</p>
                : listing.experiment?.status.includes("fallback") && <p>加速暂时不可用，已按兴趣与热度推荐。</p>}
              <button disabled={busy} onClick={() => load("/api/feed?cold=true")}>根据最新阅读更新</button>
            </div>}
            {error && (
              <div role="alert" className="empty">
                {error}
                <button onClick={() => load(listingUrl.current)}>重试</button>
              </div>
            )}
            {busy && <div className="loading">{listing.items.length ? "正在更新，以下保留上一列表…" : "正在寻找新的信号…"}</div>}
            {listing.items.length ? (
              listing.items.map((item) => (
                <Card
                  key={listing.request_id + item.id}
                  item={item}
                  request={listing.request_id}
                  onBookmark={(id, saved) =>
                    setListing((l) =>
                      l.request_id !== listing.request_id
                        ? l
                        : {
                            ...l,
                            items: l.items.map((i) =>
                              i.id === id ? { ...i, bookmarked: saved } : i,
                            ),
                          },
                    )
                  }
                />
              ))
            ) : !busy && (
              <div className="empty">
                {tab === "bookmarks"
                  ? "还没有收藏，留下你想再读的内容。"
                  : "暂时没有内容，请尝试其他关键词或稍后刷新。"}
              </div>
            )}
          </section>
          <aside>
            <div className="panel">
              <div className="eyebrow">MAKE IT YOURS</div>
              <h3>你关注什么？</h3>
              <p>选择兴趣，帮助我们找到相关内容。</p>
              <div className="topics">
                {[
                  "Artificial Intelligence",
                  "Open Source",
                  "Programming",
                  "Security",
                  "Startups",
                  "Science",
                ].map((topic) => (
                  <button
                    key={topic}
                    className={topics.includes(topic) ? "selected" : ""}
                    onClick={() => choose(topic)}
                  >
                    {topic}
                  </button>
                ))}
              </div>
            </div>
            <div className="panel source">
              <span className="live-dot" /> CONNECTED TO HACKER NEWS
              <h3>来自真实的技术社区</h3>
              <p>标题与热度来自 Hacker News。点击资讯前往原始来源。</p>
              <a
                href="https://news.ycombinator.com"
                target="_blank"
                rel="noreferrer"
              >
                浏览 Hacker News ↗
              </a>
            </div>
            <div className="panel subtle">
              <h3>这个站点正在成长</h3>
              <p>
                你的阅读与收藏将改善个人推荐。无需注册，偏好保存在当前浏览器的匿名身份下。
              </p>
              <small>
                {stats?.ctr == null
                  ? "真实效果：暂无成熟推荐曝光"
                  : `推荐成熟有效曝光 ${stats.impressions} 次 · 24h CTR ${(stats.ctr * 100).toFixed(1)}%`}
              </small>
            </div>
          </aside>
        </div>
      </main>
      <footer>
        <span>SignalFeed</span>
        <p>发现 · 阅读 · 连接</p>
        <span>Built on real signals.</span>
      </footer>
    </>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
