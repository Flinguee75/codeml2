from __future__ import annotations
import re, unicodedata, requests
from typing import Iterable, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime


PAGE_SIZE = 5000

# ---------------- Utils de base ----------------
def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.replace("œ","oe").replace("Œ","oe").replace("æ","ae").replace("Æ","ae")
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalize_address(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = strip_accents(s.strip().lower())
    # ponctuation légère
    s = re.sub(r"[.,;:']", " ", s)
    # milliers "13 555" -> "13555"
    for _ in range(3):
        s = re.sub(r"\b(\d{1,3})\s(\d{3})\b", r"\1\2", s)
    # intervalles "8400-8420" -> "8400"
    s = re.sub(r"\b(\d+)\s*-\s*\d+\b", r"\1", s)
    # abréviations fréquentes
    repl = [
        (r"\bav(?:e)?\.?\b", "avenue"), (r"\bave\b", "avenue"),
        (r"\bboul\.?\b", "boulevard"), (r"\bbd\b", "boulevard"), (r"\bblvd\b", "boulevard"),
        (r"\br\b", "rue"), (r"\brue\b", "rue"),
        (r"\bche?m(?:in)?\.?\b", "chemin"), (r"\bch\.?\b", "chemin"),
        (r"\bpl\.?\b", "place"),
        (r"\bste\b", "sainte"), (r"\bst\b", "saint"),
        (r"\bcirct\b", "circuit"), (r"\bcir\b", "circuit"),
    ]
    for pat, rep in repl: s = re.sub(pat, rep, s)
    # directions
    s = re.sub(r"(?:\s|-)\b(e)\b", " est", s)
    s = re.sub(r"(?:\s|-)\b(o)\b", " ouest", s)
    s = re.sub(r"(?:\s|-)\b(n)\b", " nord", s)
    s = re.sub(r"(?:\s|-)\b(s)\b", " sud", s)
    # unités (#, app, etc.)
    s = re.sub(r"\b(app|apt|suite|ste|local|bureau|unite|unit|#)\b[\s\-]*\w+", " ", s)
    s = re.sub(r"#\s*\w+", " ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def coerce_year_any(series: pd.Series) -> pd.Series:
    y = (series.astype(str).str.extract(r'(\d{4})')[0].pipe(pd.to_numeric, errors='coerce'))
    this_year = datetime.now().year
    y = y.where((y >= 1800) & (y <= this_year + 1))
    return y.astype("Int64")

def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip().str.replace(",", ".", regex=False),
                         errors="coerce")

def pick_first_nonnull(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([pd.NA]*len(df), index=df.index, dtype="float")
    parsed = df[present].apply(to_number, axis=0)
    return parsed.bfill(axis=1).iloc[:, 0]

# ---------------- API CKAN dynamique (sans cache) ----------------

def fetch_ckan_resource(url: str, page_size: int = PAGE_SIZE, timeout: int = 60) -> pd.DataFrame:
    frames, offset = [], 0
    while True:
        r = requests.get(url, params={"limit": page_size, "offset": offset}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        recs = data.get("result", {}).get("records", [])
        if not recs:
            break
        frames.append(pd.DataFrame(recs))
        if len(recs) < page_size:
            break
        offset += page_size
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def combine_energy_apis(api_urls: list[str], default_years: list[int|None]) -> pd.DataFrame:
    parts = []
    for url, def_year in zip(api_urls, default_years):
        rid = url.split("resource_id=")[-1][:8]
        df_i = fetch_ckan_resource(url)
        if df_i.empty:
            continue
        df_i = df_i.copy()
        df_i["source_api"] = rid
        # Année
        year_col = "Annee_consommation"
        if year_col not in df_i.columns:
            for c in ["annee", "Annee", "annee_consommation", "Annee_conso"]:
                if c in df_i.columns:
                    df_i[year_col] = df_i[c]
                    break
        df_i[year_col] = coerce_year_any(df_i.get(year_col, pd.Series([pd.NA]*len(df_i))))
        if def_year is not None:
            df_i[year_col] = df_i[year_col].fillna(def_year)
        parts.append(df_i)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

# ---------------- Colonnes métiers ----------------

def ensure_address_columns(df: pd.DataFrame, addr_col: str = "Adresse_civique") -> pd.DataFrame:
    if addr_col not in df.columns:
        raise ValueError(f"Adresse column '{addr_col}' not found.")
    df[addr_col] = df[addr_col].astype(str).str.strip()
    df["Adresse_norm"] = df[addr_col].apply(normalize_address)
    return df

def make_ges_column(df: pd.DataFrame) -> pd.DataFrame:
    candidates = ["Emissions_GES", "Emissions_GES (tCO2e)"]
    df["GES_tCO2e"] = pick_first_nonnull(df, candidates)
    return df

def make_consumption_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Electricite_GJ": ["Electricite_GJ", "Electricité_GJ", "Electricite (GJ)", "Electricite"],
        "Gaz_naturel_GJ": ["Gaz_naturel_GJ", "Gaz naturel_GJ", "Gaz_naturel (GJ)", "Gaz_naturel"],
        "Mazout_GJ": ["Mazout_GJ", "Mazout (GJ)", "Mazout"],
    }
    for out_col, candidates in mapping.items():
        present = [c for c in candidates if c in df.columns]
        df[out_col] = pick_first_nonnull(df, present) if present else pd.NA
    return df

# ---------------- Agrégations ----------------

def aggregate_by_year(df: pd.DataFrame, years: Iterable[int], year_col: str = "Annee_consommation") -> pd.DataFrame:
    keep = ["Adresse_norm", year_col, "GES_tCO2e", "Electricite_GJ", "Gaz_naturel_GJ", "Mazout_GJ"]
    keep = [c for c in keep if c in df.columns]
    g = (df[keep]
         .groupby(["Adresse_norm", year_col], as_index=False)
         .mean(numeric_only=True))

    out = {"Adresse_norm": g["Adresse_norm"].drop_duplicates().reset_index(drop=True)}
    years = list(years)

    if "GES_tCO2e" in g.columns:
        p = g.pivot(index="Adresse_norm", columns=year_col, values="GES_tCO2e")
        for y in years:
            out[str(y)] = p[y].reset_index(drop=True) if y in p.columns else pd.NA

    out_df = pd.DataFrame(out)

    for col, tag in [("Electricite_GJ", "elec"), ("Gaz_naturel_GJ", "gaz"), ("Mazout_GJ", "mazout")]:
        if col in g.columns:
            piv = g.pivot(index="Adresse_norm", columns=year_col, values=col)
            for y in years:
                out_df[f"{tag}_{y}"] = piv[y].reset_index(drop=True) if y in piv.columns else pd.NA
    return out_df

def mean_last_n(row, cols: list[str], n: int) -> float:
    vals = row[cols].dropna()
    return (vals.tail(n).mean() if len(vals) else pd.NA)

def build_core_table(energy_df: pd.DataFrame, ges_pivot: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    base_cols = ["Adresse_norm", "Adresse_civique", "Annee_construction", "Superficie", "Superficie_m2"]
    base_cols = [c for c in base_cols if c in energy_df.columns]
    core = (energy_df[base_cols]
            .dropna(subset=["Adresse_norm"])
            .drop_duplicates(subset=["Adresse_norm"], keep="first"))
    if "Superficie" in core.columns and "Superficie_m2" not in core.columns:
        core = core.rename(columns={"Superficie": "Superficie_m2"})
    ycols = [str(y) for y in years if str(y) in ges_pivot.columns]
    core = core.merge(ges_pivot[["Adresse_norm"] + ycols], on="Adresse_norm", how="left")
    core["GES_moyenne_5_ans"] = core.apply(lambda r: mean_last_n(r, ycols, 5), axis=1)
    return core

# ---------------- Mapping exact des IDs municipaux ----------------

# ==========================================
# Mapping "legacy" (exact -> exact_norm -> fuzzy limité)
# ==========================================
def map_municipal_id(
    core_base: pd.DataFrame,
    batiments_df: pd.DataFrame,
    addr_candidates_buildings = ("Adresse_civique","adresse_civique","Adresse","address","ADRESSE"),
    id_candidates   = ("building_id","buildingid","ID","id","Identifiant","id_batiment"),
    enable_fuzzy_strict: bool = True,
    enable_fuzzy_loose: bool = True,
    thresh_strict: int = 90,
    thresh_loose: int = 85,
) -> pd.DataFrame:
    """
    Reproduit la logique du notebook d'origine:
      1) exact sur (civic_num + street_key)
      2) exact sur Adresse_norm
      3) fuzzy token_set limité au même numéro civique (strict >=90, puis loose >=85)
    - Résolution d'ambiguïtés par mode() de l'ID côté municipal
    - Retourne 1 ligne / Adresse_norm (priorité: exact_key > exact_norm > fuzzy_strict > fuzzy_loose)
    """
    
    # -- colonnes adresse/ID du dataset municipal
    def _pick_first(cands, cols):
        for c in cands:
            if c in cols: return c
        return None
    
    def _split_civic_and_street(address: str):
        s = normalize_address(address)
        toks = re.split(r"[ \-]", s)
        toks = [t for t in toks if t]
        civic_num = None
        rest = toks
        if toks and re.fullmatch(r"\d+[a-z]?", toks[0]):
            m = re.match(r"(\d+)", toks[0])
            civic_num = int(m.group(1)) if m else None
            rest = toks[1:]
        STOPWORDS = {"r","rue","avenue","av","ave","boulevard","bd","boul","chemin","ch","place","est","ouest","nord","sud","montreal","quebec","canada"}
        rest = [t for t in rest if t not in STOPWORDS]
        street_key = " ".join(sorted(rest))
        return civic_num, street_key

    addr_b_col = _pick_first(addr_candidates_buildings, list(batiments_df.columns))
    id_b_col   = _pick_first(id_candidates, list(batiments_df.columns))
    if addr_b_col is None or id_b_col is None:
        raise ValueError("Colonnes adresse/ID manquantes côté municipal.")

    # -- table LEFT (énergie) unique par Adresse_norm
    left = core_base.copy()
    if "Adresse_norm" not in left.columns:
        if "Adresse_civique" in left.columns:
            left["Adresse_norm"] = left["Adresse_civique"].astype(str).apply(normalize_address)
        else:
            raise ValueError("core_base doit contenir 'Adresse_norm' ou 'Adresse_civique'.")

    left = (left
            .dropna(subset=["Adresse_norm"])
            .drop_duplicates("Adresse_norm", keep="first")
            .copy())

    left[["civic_num","street_key"]] = left["Adresse_norm"].apply(lambda s: pd.Series(_split_civic_and_street(s)))
    left["exact_key"] = left.apply(
        lambda r: f"{r['street_key']}|{int(r['civic_num'])}" if (pd.notna(r["civic_num"]) and r["street_key"]) else None,
        axis=1
    )

    # -- table RIGHT (municipal)
    right = batiments_df[[addr_b_col, id_b_col]].dropna(subset=[addr_b_col]).copy()
    right["Adresse_norm_b"] = right[addr_b_col].astype(str).apply(normalize_address)
    right[["civic_num","street_key"]] = right["Adresse_norm_b"].apply(lambda s: pd.Series(_split_civic_and_street(s)))
    right["exact_key"] = right.apply(
        lambda r: f"{r['street_key']}|{int(r['civic_num'])}" if (pd.notna(r["civic_num"]) and r["street_key"]) else None,
        axis=1
    )

    # -- résolution d'ambiguïtés: mode(ID) par exact_key et par Adresse_norm_b
    ref_exact = (right.dropna(subset=["exact_key"])
                      .groupby("exact_key", as_index=False)[id_b_col]
                      .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
                      .rename(columns={id_b_col: "municipal_id"}))

    ref_norm  = (right.groupby("Adresse_norm_b", as_index=False)[id_b_col]
                      .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
                      .rename(columns={"Adresse_norm_b":"Adresse_norm", id_b_col:"municipal_id"}))

    # -- résultat
    out = left[["Adresse_norm"]].copy()
    out["municipal_id"] = pd.NA
    out["match_method"] = pd.NA
    out["match_score"]  = pd.NA

    # 1) exact_key
# 1) exact_key
    tmp = left.merge(ref_exact, on="exact_key", how="left")[["municipal_id"]]
    tmp.index = left.index  # <<< aligne l'index sur 'left' (et donc 'out')

    m = tmp["municipal_id"].notna()
    assign_df = tmp.loc[m, ["municipal_id"]].assign(match_method="exact_key", match_score=100)
    out.loc[m, ["municipal_id","match_method","match_score"]] = assign_df.values



    # 2) exact_norm
    # 2) exact_norm
    remaining = out["municipal_id"].isna()

    # on merge uniquement les lignes restantes, puis on ALIGNE l'index sur celles d'out
    tmp2 = (
        left.loc[remaining, ["Adresse_norm"]]
            .merge(ref_norm, on="Adresse_norm", how="left")[["municipal_id"]]
    )
    tmp2.index = out.index[remaining]  # <<< index aligné avec 'out'

    # mask sur tmp2 (même index que les lignes 'remaining' de out)
    m2 = tmp2["municipal_id"].notna()

    # lignes exactes à écrire + indices correspondants dans 'out'
    assign_df2 = tmp2.loc[m2, ["municipal_id"]].assign(
        match_method="exact_norm",
        match_score=100
    )
    idx_where = tmp2.index[m2]  # <<< indices cibles dans 'out'

    # écriture alignée
    out.loc[idx_where, ["municipal_id", "match_method", "match_score"]] = assign_df2.values
        


    # 3) fuzzy token_set limité au même numéro civique (strict puis loose)
    try:
        from rapidfuzz import fuzz
        def _score(a,b): return fuzz.token_set_ratio(a,b)
    except Exception:
        import difflib
        def _score(a,b):
            # token_set approximatif
            A = " ".join(sorted(set(a.split())))
            B = " ".join(sorted(set(b.split())))
            return int(100 * difflib.SequenceMatcher(None, A, B).ratio())

    # index par numéro civique côté RIGHT
    civic_groups = {
        k: grp[["Adresse_norm_b","street_key",id_b_col]].copy().reset_index(drop=True)
        for k, grp in right.dropna(subset=["civic_num"]).groupby("civic_num")
    }

    def _best_in_group(q_street_key: str, grp_df: pd.DataFrame):
        if not q_street_key or grp_df.empty:
            return (None, None)
        target = " ".join(sorted(set(q_street_key.split())))
        best_id, best_sc = None, -1
        for _, row in grp_df.iterrows():
            cand = " ".join(sorted(set(str(row["street_key"]).split())))
            sc = _score(target, cand)
            if sc > best_sc:
                best_sc = sc
                best_id = row[id_b_col]
        return best_id, best_sc

    def _apply_fuzzy(threshold: int, tag: str):
        rem = out["municipal_id"].isna()
        if not rem.any():
            return
        # civil / rue pour LEFT
        aux = left.loc[rem, ["Adresse_norm","civic_num","street_key"]].copy()
        idxs = aux.index.tolist()
        for i in idxs:
            cn = aux.at[i, "civic_num"]
            sk = aux.at[i, "street_key"]
            if pd.isna(cn) or cn not in civic_groups:
                continue
            bid, sc = _best_in_group(sk, civic_groups[cn])
            if bid is not None and sc is not None and sc >= threshold:
                out.at[i, "municipal_id"] = bid
                out.at[i, "match_method"] = tag
                out.at[i, "match_score"]  = int(sc)

    if enable_fuzzy_strict:
        _apply_fuzzy(thresh_strict, "fuzzy_strict")
    if enable_fuzzy_loose:
        _apply_fuzzy(thresh_loose, "fuzzy_loose")

    # 1 ligne par Adresse_norm (priorité déjà respectée par l'ordre des passes)
    out = out.drop_duplicates(subset=["Adresse_norm"], keep="first").reset_index(drop=True)

    return out


# -------- Consommations + moyennes 5 ans + exports (format ref) --------

def build_core_table_with_consumption(energy_df: pd.DataFrame,
                                      ges_pivot: pd.DataFrame,
                                      years: list[int]) -> pd.DataFrame:
    base_cols = ["Adresse_norm", "Adresse_civique", "Annee_construction", "Superficie", "Superficie_m2"]
    base_cols = [c for c in base_cols if c in energy_df.columns]

    core = (
        energy_df[base_cols]
        .dropna(subset=["Adresse_norm"])
        .drop_duplicates(subset=["Adresse_norm"], keep="first")
        .copy()
    )

    if "Superficie" in core.columns and "Superficie_m2" not in core.columns:
        core = core.rename(columns={"Superficie": "Superficie_m2"})

    ges_year_cols = [str(y) for y in years if str(y) in ges_pivot.columns]
    cons_cols = []
    for y in years:
        for prefix in ("elec_", "gaz_", "mazout_"):
            col = f"{prefix}{y}"
            if col in ges_pivot.columns:
                cons_cols.append(col)

    cols_to_merge = ["Adresse_norm"] + ges_year_cols + cons_cols
    core = core.merge(ges_pivot[cols_to_merge], on="Adresse_norm", how="left")

    def _mean_last5(row):
        vals = row[ges_year_cols].dropna()
        if len(vals) == 0:
            return pd.NA
        ordered = [c for c in ges_year_cols if c in row.index]
        return row[ordered].astype(float).mean()

    core["GES_moyenne_5_ans"] = core.apply(_mean_last5, axis=1)
    return core


def finalize_export(core_with_id: pd.DataFrame,
                    core_base_with_cons: pd.DataFrame,
                    years: list[int]) -> pd.DataFrame:
    final = core_with_id.merge(core_base_with_cons, on="Adresse_norm", how="left")
    if "municipal_id" in final.columns and "building_id" not in final.columns:
        final = final.rename(columns={"municipal_id": "building_id"})

    fixed = ["building_id", "Adresse_civique", "Adresse_norm", "Annee_construction", "Superficie_m2", "GES_moyenne_5_ans"]

    cons_cols = []
    for y in years:
        for prefix in ("elec_", "gaz_", "mazout_"):
            col = f"{prefix}{y}"
            if col in final.columns:
                cons_cols.append(col)

    ordered = [c for c in fixed if c in final.columns] + cons_cols
    final = final[[c for c in ordered if c in final.columns]].copy()

    if "Adresse_civique" in final.columns:
        final = final.sort_values("Adresse_civique", kind="stable").reset_index(drop=True)

    return final

def build_5y_means(core_base_with_years: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    df = core_base_with_years.copy()

    def _num_df(dfa: pd.DataFrame) -> pd.DataFrame:
        return dfa.apply(lambda s: pd.to_numeric(s, errors="coerce")).astype("float64")

    ges_year_cols = [str(y) for y in years if str(y) in df.columns]
    if ges_year_cols:
        df["GES_moyenne_5_ans"] = _num_df(df[ges_year_cols]).mean(axis=1, skipna=True)
    else:
        df["GES_moyenne_5_ans"] = np.nan

    def _mean_cols(prefix: str) -> pd.Series:
        cols = [f"{prefix}{y}" for y in years if f"{prefix}{y}" in df.columns]
        if not cols:
            return pd.Series(np.nan, index=df.index, dtype="float64")
        return _num_df(df[cols]).mean(axis=1, skipna=True).astype("float64")

    df["Electricite_GJ_moyenne_5_ans"] = _mean_cols("elec_")
    df["Gaz_naturel_GJ_moyenne_5_ans"] = _mean_cols("gaz_")
    df["Mazout_GJ_moyenne_5_ans"]      = _mean_cols("mazout_")

    df["Conso_totale_GJ"] = pd.concat(
        [
            df["Electricite_GJ_moyenne_5_ans"],
            df["Gaz_naturel_GJ_moyenne_5_ans"],
            df["Mazout_GJ_moyenne_5_ans"],
        ],
        axis=1,
    ).astype("float64").sum(axis=1, skipna=True)

    return df


def finalize_export_5y_means(core_with_id: pd.DataFrame,
                             core_base_with_years: pd.DataFrame,
                             years: list[int]) -> pd.DataFrame:
    merged = core_with_id.drop_duplicates("Adresse_norm", keep="first") \
                     .merge(core_base_with_years.drop_duplicates("Adresse_norm", keep="first"),
                            on="Adresse_norm", how="left")

    if "municipal_id" in merged.columns and "building_id" not in merged.columns:
        merged = merged.rename(columns={"municipal_id": "building_id"})

    means = build_5y_means(merged, years=years)
    merged = merged.join(means[["GES_moyenne_5_ans",
                                "Electricite_GJ_moyenne_5_ans",
                                "Gaz_naturel_GJ_moyenne_5_ans",
                                "Mazout_GJ_moyenne_5_ans",
                                "Conso_totale_GJ"]])

    if "Superficie_m2" in merged.columns:
        merged["Superficie"] = pd.to_numeric(merged["Superficie_m2"], errors="coerce")
    elif "Superficie" in merged.columns:
        merged["Superficie"] = pd.to_numeric(merged["Superficie"], errors="coerce")
    else:
        merged["Superficie"] = pd.NA

    merged["GES"] = pd.to_numeric(merged["GES_moyenne_5_ans"], errors="coerce")
    merged["GES_m2"] = merged["GES"] / merged["Superficie"]

    out = pd.DataFrame()
    out["id"]                              = merged.get("building_id")
    out["Adresse"]                         = merged.get("Adresse_civique")
    out["Annee_construction"]              = merged.get("Annee_construction")
    out["Superficie"]                      = merged.get("Superficie")
    out["GES"]                             = merged.get("GES")
    out["Electricite_GJ_moyenne_5_ans"]    = merged.get("Electricite_GJ_moyenne_5_ans")
    out["Gaz_naturel_GJ_moyenne_5_ans"]    = merged.get("Gaz_naturel_GJ_moyenne_5_ans")
    out["Mazout_GJ_moyenne_5_ans"]         = merged.get("Mazout_GJ_moyenne_5_ans")
    out["Conso_totale_GJ"]                 = merged.get("Conso_totale_GJ")
    out["GES_m2"]                          = merged.get("GES_m2")

    if "Adresse" in out.columns:
        out = out.sort_values("Adresse", kind="stable").reset_index(drop=True)

    return out


