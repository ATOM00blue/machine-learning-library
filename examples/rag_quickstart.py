#!/usr/bin/env python3
"""Minimal semantic-search / RAG quickstart over the ML library corpus.

Embeds every document (chunked by Markdown heading) into a local LanceDB table,
then answers a query by returning the most relevant chunks WITH their source URL
so you can cite. This is the "true semantic retrieval" path for the 10M-token
corpus — use it when keyword/grep search isn't enough.

Usage:
    pip install lancedb sentence-transformers python-frontmatter tqdm
    python examples/rag_quickstart.py --build                  # one-time index
    python examples/rag_quickstart.py "how does rotary positional encoding work?"

It's intentionally tiny (~80 lines) and dependency-light. Swap the embedding
model, chunker, or vector store for your own stack — this is a starting point,
not a framework.
"""
from __future__ import annotations
import argparse, glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")
DB_DIR = os.path.join(os.path.dirname(__file__), ".lancedb")
TABLE = "ml_library"
MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, fast, strong on technical text
MAX_CHUNK_CHARS = 2000


def iter_chunks():
    """Yield (text, metadata) chunks, split on Markdown headings."""
    import frontmatter
    for path in glob.glob(os.path.join(CORPUS, "**", "*.md"), recursive=True):
        if os.path.basename(path) == "INDEX.md":
            continue
        post = frontmatter.load(path)
        meta = post.metadata
        title = meta.get("title", os.path.basename(path))
        url = meta.get("url", "")
        # split body on headings, then hard-wrap long sections
        parts = re.split(r"\n(?=#{1,6}\s)", post.content)
        for part in parts:
            part = part.strip()
            if len(part) < 40:
                continue
            for i in range(0, len(part), MAX_CHUNK_CHARS):
                chunk = part[i:i + MAX_CHUNK_CHARS]
                yield chunk, dict(title=str(title), url=str(url),
                                  source=str(meta.get("source", "")),
                                  path=os.path.relpath(path, ROOT))


def build():
    import lancedb
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
    model = SentenceTransformer(MODEL)
    chunks, metas = [], []
    for text, meta in tqdm(iter_chunks(), desc="reading"):
        chunks.append(text); metas.append(meta)
    print(f"embedding {len(chunks)} chunks with {MODEL} ...")
    vecs = model.encode(chunks, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True)
    rows = [dict(vector=v, text=t, **m) for v, t, m in zip(vecs.tolist(), chunks, metas)]
    db = lancedb.connect(DB_DIR)
    db.create_table(TABLE, data=rows, mode="overwrite")
    print(f"indexed {len(rows)} chunks -> {DB_DIR}")


def query(q: str, k: int = 5):
    import lancedb
    from sentence_transformers import SentenceTransformer
    db = lancedb.connect(DB_DIR)
    if TABLE not in db.table_names():
        sys.exit("No index yet. Run:  python examples/rag_quickstart.py --build")
    model = SentenceTransformer(MODEL)
    qv = model.encode(q, normalize_embeddings=True)
    hits = db.open_table(TABLE).search(qv).limit(k).to_list()
    print(f"\nTop {k} chunks for: {q!r}\n" + "=" * 60)
    for h in hits:
        print(f"\n## {h['title']}\n{h['url']}\n({h['path']})\n")
        print(h["text"][:700].strip(), "...")
    print("\n" + "=" * 60)
    print("Cite each answer with the url shown above.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", nargs="*", help="natural-language question")
    ap.add_argument("--build", action="store_true", help="(re)build the index")
    ap.add_argument("-k", type=int, default=5, help="number of chunks to return")
    args = ap.parse_args()
    if args.build:
        build()
    if args.query:
        query(" ".join(args.query), args.k)
    elif not args.build:
        ap.print_help()


if __name__ == "__main__":
    main()
