"""
Electronics Technology Intelligence — Demo UI
Run with: streamlit run streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Electronics Intelligence",
    page_icon="🤴",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Cached loaders (persist across reruns)
@st.cache_resource
def load_rag_app():
    from app.rag.graph import app as rag_app
    return rag_app

@st.cache_resource
def load_chroma_client():
    import chromadb
    from app.config.settings import PATHS
    return chromadb.PersistentClient(path=str(PATHS["chroma"]))

@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    from app.config.settings import EMBEDDING_MODEL
    return SentenceTransformer(EMBEDDING_MODEL)

#  Sidebar
with st.sidebar:
    st.title("Electronics Intel")
    st.caption("RAG-powered technology intelligence")

    st.divider()
    st.subheader(" Index Status")
    try:
        client = load_chroma_client()
        news_c = client.get_collection("news_articles").count()
        dict_c = client.get_collection("dictionaries").count()
        col1, col2 = st.columns(2)
        col1.metric("News", f"{news_c:,}")
        col2.metric("Dict", f"{dict_c:,}")
    except Exception as e:
        st.warning(f"Vector store not available: {e}")

    st.divider()
    st.subheader("Feeds")
    try:
        from app.config.settings import FEEDS
        enabled = sum(1 for f in FEEDS if f.enabled)
        st.write(f"**{enabled}** / {len(FEEDS)} enabled")
        with st.expander("View all"):
            for f in FEEDS:
                icon = "🟢" if f.enabled else "⚪"
                st.write(f"{icon} {f.source_name}")
    except Exception as e:
        st.warning(f"Could not load feeds: {e}")

    st.divider()
    if st.button("🔄 Reload caches", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

#  Main 
st.title(" Electronics Technology Intelligence")
st.caption("Ask a technical question. The RAG pipeline retrieves from your dictionaries "
           "and news corpus, then generates a grounded summary.")

tab_query, tab_explore, tab_bandit, tab_feeds = st.tabs(
    [" Query", " Explore Corpus", " Bandit Demo", " Feeds"]
)

#TAB 1: Query 
with tab_query:
    examples = [
        "What is impedance matching?",
        "Latest advances in 6G terahertz communication",
        "Explain GaN power electronics",
        "Define ripple in rectifiers",
    ]
    st.write("**Try an example:**")
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["query_input"] = ex

    query = st.text_input(
        "Your question",
        key="query_input",
        placeholder="e.g. How does a Smith chart work?",
    )

    run = st.button(" Run RAG", type="primary", use_container_width=True)

    if run and query.strip():
        with st.spinner("Retrieving context and generating summary…"):
            try:
                rag_app = load_rag_app()
                result = rag_app.invoke({"query": query})
                st.session_state["last_result"] = result
                st.session_state["last_query"] = query
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.exception(e)

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.divider()
        st.subheader(f" Summary — _{st.session_state['last_query']}_")
        st.markdown(result.get("summary", "_(no summary)_"))

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Sources", len(result.get("sources", [])))
        m2.metric("News chunks", len(result.get("retrieved_docs", [])))
        m3.metric("Dict chunks", len(result.get("dictionary_context", [])))

        if result.get("needs_review"):
            st.warning(f" Quality score {result.get('quality_score')} — flagged for review")

        # Sources
        sources = result.get("sources", [])
        if sources:
            with st.expander(f" Sources ({len(sources)})"):
                for s in sources:
                    st.markdown(f"- [{s}]({s})")

        # Retrieved news context
        docs = result.get("retrieved_docs", [])
        if docs:
            with st.expander(f" News context ({len(docs)})"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**{i}. {d.get('source', 'n/a')}**")
                    st.caption(d.get("content", "")[:500] + "…")

        # Dictionary context
        dctx = result.get("dictionary_context", [])
        if dctx:
            with st.expander(f" Dictionary context ({len(dctx)})"):
                for i, c in enumerate(dctx, 1):
                    st.markdown(f"**{i}.** {c[:500]}…")

# TAB 2: Explore
with tab_explore:
    st.subheader("Direct semantic search")
    col_a, col_b, col_c = st.columns([3, 1, 1])
    q = col_a.text_input("Search query", key="explore_q", placeholder="impedance matching")
    collection = col_b.selectbox("Collection", ["news_articles", "dictionaries"])
    k = col_c.number_input("Top K", 1, 20, 5)

    if st.button(" Search", use_container_width=True) and q.strip():
        try:
            client = load_chroma_client()
            col = client.get_collection(collection)
            emb = load_embedder().encode([q]).tolist()
            res = col.query(query_embeddings=emb, n_results=k)

            for i, (doc, meta, dist) in enumerate(zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            ), 1):
                title = meta.get("title") or meta.get("dictionary_name", "n/a")
                st.markdown(f"**{i}. {title}** &nbsp; _distance = {dist:.3f}_")
                st.caption(doc[:400] + ("…" if len(doc) > 400 else ""))
                st.divider()
        except Exception as e:
            st.error(f"Search failed: {e}")

#TAB 3: Bandit Demo
with tab_bandit:
    st.subheader("Contextual Bandit (LinUCB)")
    st.caption("Simulates a news recommender learning which article category a user prefers.")

    c1, c2, c3 = st.columns(3)
    n_arms = c1.slider("Categories", 2, 8, 4)
    alpha = c2.slider("Exploration (α)", 0.05, 2.0, 0.5, 0.05)
    rounds = c3.slider("Rounds", 50, 1000, 300, 50)

    true_best = st.selectbox("Simulated user's true preference (arm index)", list(range(n_arms)))

    if st.button("▶ Simulate", type="primary", use_container_width=True):
        from app.bandit.linucb import LinUCB
        bandit = LinUCB(n_arms=n_arms, context_dim=5, alpha=alpha)
        picks = np.zeros(n_arms, dtype=int)

        for _ in range(rounds):
            ctx = np.random.randn(5, 1)
            arm = bandit.select(ctx)
            picks[arm] += 1
            reward = 1.0 if arm == true_best else 0.0
            bandit.update(arm, ctx, reward)

        final_ctx = np.random.randn(5, 1)
        learned = bandit.select(final_ctx)

        st.success(
            f"After {rounds} rounds: bandit prefers arm **{learned}**, "
            f"true best is **{true_best}**"
        )
        st.bar_chart({"times picked": picks})

# TAB 4: Feeds
with tab_feeds:
    st.subheader("News ingestion")
    st.write("Fetch the latest articles from all enabled RSS feeds.")

    if st.button(" Run ingestion now", type="primary", use_container_width=True):
        with st.spinner("Fetching feeds…"):
            try:
                from app.pipeline.daily_pipeline import run_ingestion
                new = run_ingestion()
                st.success(f"✓ Ingested {len(new)} new articles")
                if new:
                    st.dataframe(
                        [{"Title": a["title"][:80], "Source": a.get("source_id", "")}
                         for a in new[:20]],
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
                st.exception(e)

    st.divider()
    st.subheader("Recent articles in corpus")
    try:
        import json
        from app.config.settings import PATHS
        path = PATHS["processed"] / "documents.jsonl"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            st.metric("Total articles in corpus", len(lines))
            recent = [json.loads(l) for l in lines[-10:]][::-1]
            for a in recent:
                st.markdown(f"- [{a['title']}]({a['url']})  \n  _{a.get('source', '')}_")
    except Exception as e:
        st.warning(f"Could not load corpus: {e}")

st.divider()
st.caption(" Electronics Technology Intelligence · RAG + LangGraph + ChromaDB")