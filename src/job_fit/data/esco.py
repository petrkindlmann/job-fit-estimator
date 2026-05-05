from dataclasses import dataclass
from pathlib import Path
import csv
import re


@dataclass
class EscoCandidate:
    isco_code: str
    label_en: str
    label_cs: str
    keywords: list[str]
    score: float = 0.0


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-záčďéěíňóřšťúůýž]+", (text or "").lower()))


class EscoIndex:
    DEFAULT_PATH = Path("data/esco/occupations.csv")

    def __init__(self, candidates: list[EscoCandidate]):
        self.candidates = candidates
        self._token_sets = [_tokenize(c.label_en + " " + c.label_cs + " " + " ".join(c.keywords))
                            for c in candidates]

    @classmethod
    def load_default(cls) -> "EscoIndex":
        return cls.load(cls.DEFAULT_PATH)

    @classmethod
    def load(cls, path: Path) -> "EscoIndex":
        cands = []
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kws = [k.strip() for k in (row["keywords_en"] + "," + row["keywords_cs"]).split(",") if k.strip()]
                cands.append(EscoCandidate(
                    isco_code=row["isco_code"],
                    label_en=row["label_en"],
                    label_cs=row["label_cs"],
                    keywords=kws,
                ))
        return cls(cands)

    def search(self, query: str, k: int = 5) -> list[EscoCandidate]:
        q_tokens = _tokenize(query)
        scored: list[EscoCandidate] = []
        for cand, ts in zip(self.candidates, self._token_sets):
            if not ts:
                continue
            overlap = len(q_tokens & ts)
            if overlap == 0:
                continue
            # Jaccard-ish + bonus for keyword exact substring match
            jaccard = overlap / max(1, len(q_tokens | ts))
            substring_bonus = 0.0
            ql = query.lower()
            for kw in cand.keywords:
                if kw.lower() in ql:
                    substring_bonus += 0.1
            scored_cand = EscoCandidate(
                isco_code=cand.isco_code, label_en=cand.label_en, label_cs=cand.label_cs,
                keywords=cand.keywords, score=jaccard + substring_bonus,
            )
            scored.append(scored_cand)
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]

    def by_code(self, isco: str) -> EscoCandidate | None:
        for c in self.candidates:
            if c.isco_code == isco:
                return c
        return None
