from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd


@dataclass
class IspvRecord:
    isco_code: str
    isco_level: int    # 2, 3, or 4
    d1: float
    q1: float
    median: float
    q3: float
    d9: float
    mean: float
    period: str        # e.g. "rok 2025"
    sphere: str        # "MZDOVA" | "PLATOVA"
    count: int


# Actual MPSV keys confirmed from live JSON (rok 2025 dataset).
# Each list is tried in order; first match wins.
FIELD_MAP = {
    "isco":   ["czIsco", "cz_isco", "isco_code", "isco"],
    "d1":     ["diferenciaceD1M", "d1", "decile_1"],
    "q1":     ["diferenciaceQ1M", "q1", "quartile_1"],
    "median": ["medianMzda", "medianPlat", "median"],
    "q3":     ["diferenciaceQ3M", "q3", "quartile_3"],
    "d9":     ["diferenciaceD9M", "d9", "decile_9"],
    "mean":   ["mzdaPrumer", "platPrumer", "mean", "average"],
    "period": ["obdobi", "period"],
    "sphere": ["sfera", "sphere"],
    "count":  ["pocetZamestnancuMzda", "pocetZamestnancu", "employee_count", "count"],
}


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    resolved = {}
    for canon, candidates in FIELD_MAP.items():
        for c in candidates:
            if c in df.columns:
                resolved[canon] = c
                break
    missing = set(FIELD_MAP) - set(resolved)
    if missing:
        hard_missing = missing - {"sphere", "count", "period"}
        if hard_missing:
            raise RuntimeError(f"ISPV columns not found: {hard_missing}")
    return resolved


def _normalize_isco(code: str) -> str:
    """Strip 'CzIsco/' prefix if present, keep only digits."""
    s = str(code).split("/")[-1]
    return "".join(ch for ch in s if ch.isdigit())


class IspvIndex:
    DEFAULT_PATH = Path("data/ispv/ispv.parquet")

    def __init__(self, df: pd.DataFrame):
        cols = _resolve_columns(df)
        self.df = df.copy()
        self.df["_isco"] = self.df[cols["isco"]].astype(str).map(_normalize_isco)
        self.df["_isco_level"] = self.df["_isco"].str.len()
        self._cols = cols
        # Prefer the most recent period; if no period column, keep all rows.
        if "period" in cols:
            most_recent = self.df[cols["period"]].mode()
            if len(most_recent) > 0:
                self.df = self.df[self.df[cols["period"]] == most_recent.iloc[0]]
        self._by_isco = {row["_isco"]: row for _, row in self.df.iterrows()}

    @classmethod
    def load_default(cls) -> "IspvIndex":
        return cls.load(cls.DEFAULT_PATH)

    @classmethod
    def load(cls, path: Path) -> "IspvIndex":
        df = pd.read_parquet(path)
        return cls(df)

    def iscos(self) -> set[str]:
        return set(self._by_isco.keys())

    def _row_to_record(self, row) -> IspvRecord:
        c = self._cols
        count_val = row[c["count"]] if "count" in c else None
        return IspvRecord(
            isco_code=row["_isco"],
            isco_level=int(row["_isco_level"]),
            d1=float(row[c["d1"]]),
            q1=float(row[c["q1"]]),
            median=float(row[c["median"]]),
            q3=float(row[c["q3"]]),
            d9=float(row[c["d9"]]),
            mean=float(row[c["mean"]]),
            period=str(row[c["period"]]) if "period" in c else "",
            sphere=str(row[c["sphere"]]) if "sphere" in c else "",
            count=int(count_val) if "count" in c and pd.notna(count_val) else 0,
        )

    def lookup(self, isco: str) -> Optional[IspvRecord]:
        isco_n = _normalize_isco(isco)
        row = self._by_isco.get(isco_n)
        if row is None:
            return None
        return self._row_to_record(row)

    def lookup_with_rollup(self, isco: str) -> Optional[IspvRecord]:
        """Try 4-digit, then 3-digit, then 2-digit prefixes. Aggregate when needed."""
        isco_n = _normalize_isco(isco)
        for level in (4, 3, 2):
            prefix = isco_n[:level]
            exact = self._by_isco.get(prefix)
            if exact is not None:
                return self._row_to_record(exact)
            matches = [r for k, r in self._by_isco.items() if k.startswith(prefix)]
            if matches:
                df_match = pd.DataFrame(matches)
                c = self._cols
                count_col = c.get("count")
                period_col = c.get("period")
                sphere_col = c.get("sphere")
                return IspvRecord(
                    isco_code=prefix,
                    isco_level=level,
                    d1=float(df_match[c["d1"]].mean()),
                    q1=float(df_match[c["q1"]].mean()),
                    median=float(df_match[c["median"]].mean()),
                    q3=float(df_match[c["q3"]].mean()),
                    d9=float(df_match[c["d9"]].mean()),
                    mean=float(df_match[c["mean"]].mean()),
                    period=str(df_match[period_col].iloc[0]) if period_col else "",
                    sphere="AGGREGATE",
                    count=int(df_match[count_col].sum()) if count_col else 0,
                )
        return None
