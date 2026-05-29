#!/usr/bin/env python3
"""Write verified, web-fetched docs into the corpus in the standard format.

Input: a JSON file (argv[1]) = list of verified-doc objects, each:
  {source:"arxiv"|"web", url, title, authors:[...], date:"YYYY-MM[-DD]",
   arxiv_id?, category?, domain?, body_markdown, topics:[slug...],
   techniques:[...], level:"..."}

For each: write corpus/papers/<arxiv_id>.md (arxiv) or
corpus/web/<domainslug>/<titleslug>.md (web) with the same frontmatter schema as
the rest of the corpus, and record its controlled tags in tools/overrides.json so
tag_corpus.py assigns exactly those tags. Skips anything whose file already
exists (dedup). Does NOT run the tag/index/atlas pipeline — caller does that.
"""
from __future__ import annotations
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")
OV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overrides.json")
FETCHED_AT = "2026-05-29T00:00:00Z"
SLUGS = {"neural-network-foundations","classical-ml","computer-vision","sequence-models-rnn",
         "transformers-attention","language-models","efficient-architectures","generative-models",
         "multimodal","reinforcement-learning","alignment-rlhf","reasoning-agents",
         "efficiency-systems","interpretability","evaluation-trust","ml-engineering","ai-industry-news"}


def slugify(s, n=80):
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:n].strip("-") or "untitled"


def jarr(xs):
    return "[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in xs) + "]"


def main():
    docs = json.load(open(sys.argv[1]))
    overrides = json.load(open(OV_PATH)) if os.path.exists(OV_PATH) else {}
    written, skipped, bad = 0, 0, 0
    seen_arxiv = set()

    for d in docs:
        try:
            topics = [t for t in d.get("topics", []) if t in SLUGS][:4]
            if not topics or not d.get("title") or not d.get("url") or not d.get("body_markdown"):
                bad += 1; continue
            src = d["source"]
            if src == "arxiv":
                aid = (d.get("arxiv_id") or "").strip()
                if not re.match(r"^\d{4}\.\d{4,5}$", aid) or aid in seen_arxiv:
                    bad += 1; continue
                seen_arxiv.add(aid)
                rel = f"corpus/papers/{aid}.md"
                fm = ["---", f'title: {json.dumps(d["title"], ensure_ascii=False)}',
                      'source: "arxiv"', f'arxiv_id: "{aid}"', f'url: {json.dumps(d["url"])}',
                      f'authors: {jarr(d.get("authors", []))}',
                      f'published: "{d.get("date","")}"',
                      f'categories: {jarr([d.get("category","cs.LG")])}',
                      f'primary_category: "{d.get("category","cs.LG")}"',
                      f'fetched_at: "{FETCHED_AT}"',
                      f'topics: {jarr(topics)}', "---", ""]
            else:  # web
                dom = re.sub(r"[^a-z0-9]", "", (d.get("domain") or "web").lower())
                rel = f"corpus/web/{dom}/{slugify(d['title'])}.md"
                fm = ["---", f'title: {json.dumps(d["title"], ensure_ascii=False)}',
                      'source: "web"', f'url: {json.dumps(d["url"])}',
                      f'domain: "{d.get("domain","")}"',
                      f'name: {json.dumps(d["title"], ensure_ascii=False)}',
                      f'published: "{d.get("date","")}"',
                      f'fetched_at: "{FETCHED_AT}"',
                      f'topics: {jarr(topics)}', "---", ""]

            full = os.path.join(ROOT, rel)
            if os.path.exists(full):
                skipped += 1; continue
            os.makedirs(os.path.dirname(full), exist_ok=True)
            body = d["body_markdown"].strip() + "\n"
            open(full, "w", encoding="utf-8").write("\n".join(fm) + body)
            overrides[rel] = {"topics": topics,
                              "techniques": [t for t in d.get("techniques", [])][:5],
                              "level": d.get("level", "frontier")}
            written += 1
        except Exception as e:
            bad += 1
            print("  ERR:", d.get("title", "?"), e)

    json.dump(overrides, open(OV_PATH, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {written} new docs; skipped {skipped} existing; rejected {bad}")
    print(f"overrides.json now has {len(overrides)} entries")


if __name__ == "__main__":
    main()
