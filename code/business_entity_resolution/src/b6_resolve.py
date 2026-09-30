"""
b6_resolve.py - Block 6: the entity-level resolver applied on top of the pairwise accept rule.

The accept rule decides every (S1, record) pair on its own; this module reasons over GROUPS of
decisions:

  1. keys      France-aware name / address keys: accents folded, street abbreviations unified (r->rue,
               bd->boulevard, ...), leading/mid legal forms stripped (sa, sas, sarl, sci, ei, ...), house
               number found in either address order (city-first included), postal code parsed from text,
               admin-region words (departement / region) never used as street evidence.
  2. vetoes    crowding: S1 sits in a building shared by >= c1 distinct S1 business names (multi-tenant)
               -> an accept needs the NAMES to agree.  template: S1's name key is shared by >= c2 S1s
               (chain / template names like "Defense Club SARL") -> an accept is vetoed if the ADDRESSES conflict.
  3. one owner a record accepted for >= 2 S1s keeps at most one: the top S1 by probability if it leads by
               >= m1 and its evidence (name_agree + addr_agree) is not weaker than the runner-up's; else the
               S1 with clearly stronger evidence (>= e_gap) if its probability >= p_floor; else nobody.
               Train ground truth: every S2/S3 record belongs to at most one S1 (7.64M pairs, 0 exceptions).
  4. twin      recall rescue: accept a not-yet-accepted candidate of S1 s if it is a
     rescue   twin of a record s already won with p >= p_hi, its own p >= p_lo, nobody else owns it, and
               neither names nor addresses conflict.
  5. profile   label-free output profile per country (empty share, accepts per S1, collision rate) -
               France must move TOWARD India/US's profile.

Everything is deterministic and label-free. Labels (train) are used only by tune().
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd

B6_CODE_VERSION = "b6-resolve-1.0"

# --------------------------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------------------------
ABBR_FR = {"r": "rue", "bd": "boulevard", "bld": "boulevard", "blvd": "boulevard", "boul": "boulevard",
           "av": "avenue", "ave": "avenue", "aven": "avenue", "pl": "place", "ch": "chemin", "che": "chemin",
           "chem": "chemin", "rte": "route", "imp": "impasse", "all": "allee", "sq": "square", "fbg": "faubourg",
           "faub": "faubourg", "crs": "cours", "qu": "quai", "qua": "quai", "pas": "passage", "sen": "sentier",
           "res": "residence", "resid": "residence", "lot": "lotissement", "za": "zone", "zi": "zone",
           "zac": "zone", "st": "saint", "ste": "sainte", "prom": "promenade", "espl": "esplanade",
           "mte": "montee", "hameau": "hameau", "ham": "hameau", "vla": "villa", "cite": "cite", "pte": "porte"}
STREET_TYPES_FR = {"rue", "boulevard", "avenue", "place", "chemin", "route", "impasse", "allee", "square",
                   "faubourg", "cours", "quai", "passage", "sentier", "residence", "lotissement", "zone",
                   "promenade", "esplanade", "montee", "hameau", "villa", "cite", "porte", "voie", "rond",
                   "parvis", "galerie", "chaussee", "clos", "domaine", "lieu", "mail", "sente", "venelle"}
ABBR_EN = {"st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
           "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "hwy": "highway",
           "pkwy": "parkway", "sq": "square", "ter": "terrace", "cir": "circle", "trl": "trail", "wy": "way",
           "ste": "suite", "fl": "floor", "apt": "apartment", "bldg": "building", "n": "north", "s": "south",
           "e": "east", "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
           "mg": "marg", "nr": "near", "opp": "opposite", "rd.": "road"}
STREET_TYPES_EN = {"street", "road", "avenue", "boulevard", "drive", "lane", "court", "place", "highway",
                   "parkway", "square", "terrace", "circle", "trail", "way", "marg", "nagar", "colony", "sector",
                   "layout", "main", "cross", "block", "phase", "gali", "chowk", "bazar", "bazaar", "market",
                   "complex", "plaza", "tower", "towers", "mall", "estate"}
UNIT_WORDS = {"suite", "floor", "apartment", "building", "unit", "room", "bat", "batiment", "etage", "appt",
              "porte", "bureau", "lot", "escalier", "esc", "no", "num", "numero", "flat", "shop", "office"}
STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "a", "au", "aux", "en", "sur", "sous", "the", "of",
        "and", "at", "on", "in", "near", "opposite", "behind", "next", "to", "cedex", "bp", "cs", "france"}
LEGAL_FORMS = {"sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "scp", "scm", "selarl", "selas", "ei", "eirl",
               "earl", "gaec", "sca", "scop", "scea", "sel", "ste", "societe", "ets", "etablissements", "cie",
               "compagnie", "gie", "association", "asso", "llc", "inc", "ltd", "limited", "pvt", "private",
               "co", "corp", "corporation", "company", "llp", "pc", "plc", "lp", "pllc", "incorporated", "gmbh",
               "srl", "bv", "nv", "ag"}
FR_ADMIN_PHRASES = [
    "auvergne rhone alpes", "bourgogne franche comte", "centre val de loire", "provence alpes cote d azur",
    "nouvelle aquitaine", "hauts de france", "ile de france", "pays de la loire", "grand est", "champagne ardenne",
    "languedoc roussillon", "midi pyrenees", "nord pas de calais", "basse normandie", "haute normandie",
    "poitou charentes", "rhone alpes", "franche comte", "alpes de haute provence", "hautes alpes",
    "alpes maritimes", "bouches du rhone", "charente maritime", "corse du sud", "haute corse", "cote d or",
    "cotes d armor", "eure et loir", "haute garonne", "ille et vilaine", "indre et loire", "loir et cher",
    "haute loire", "loire atlantique", "lot et garonne", "maine et loire", "haute marne", "meurthe et moselle",
    "puy de dome", "pyrenees atlantiques", "hautes pyrenees", "pyrenees orientales", "bas rhin", "haut rhin",
    "haute saone", "saone et loire", "haute savoie", "seine maritime", "seine et marne", "deux sevres",
    "tarn et garonne", "haute vienne", "territoire de belfort", "hauts de seine", "seine saint denis",
    "val de marne", "val d oise", "la reunion",
    # single-word regions / departements that are NOT common street / city words
    "bretagne", "normandie", "occitanie", "corse", "alsace", "aquitaine", "auvergne", "bourgogne", "limousin",
    "lorraine", "picardie", "paca", "idf", "aisne", "allier", "ardeche", "ardennes", "ariege", "aube", "aude",
    "aveyron", "calvados", "cantal", "charente", "correze", "creuse", "dordogne", "doubs", "drome", "finistere",
    "gers", "gironde", "herault", "indre", "isere", "landes", "loiret", "lozere", "mayenne", "morbihan",
    "moselle", "nievre", "oise", "orne", "sarthe", "savoie", "somme", "tarn", "vaucluse", "vendee", "vosges",
    "yonne", "essonne", "yvelines", "guadeloupe", "martinique", "guyane", "mayotte"]
_ADMIN_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(p) for p in FR_ADMIN_PHRASES), key=len, reverse=True)) + r")\b")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_HOUSE = re.compile(r"^\d{1,4}[a-z]?$")
_ORD = {"bis", "ter", "quater", "b", "t"}


def fold(s) -> str:
    """lowercase, accents folded, non-alphanumerics -> single spaces."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return _NONALNUM.sub(" ", s).strip()


# --------------------------------------------------------------------------------------------
# 1. Keys
# --------------------------------------------------------------------------------------------
def parse_name(name) -> tuple[str, frozenset, str]:
    """(name_key, token set, first token). Legal forms stripped ANYWHERE (leading 'sa', mid 'ei', ...)."""
    toks = [t for t in fold(name).split() if t not in LEGAL_FORMS and t not in STOP]
    if not toks:
        return "", frozenset(), ""
    return " ".join(toks), frozenset(toks), toks[0]


def parse_addr(addr, country: str, postal_hint=None) -> tuple[str, str, frozenset, bool]:
    """(postal, house_number, street token set, strong) for one address.
    strong = a street-type word was found (rue/avenue/road/...), so street tokens are reliable."""
    s = fold(addr)
    fr = str(country).upper() == "FRANCE"
    if fr:
        s = _ADMIN_RE.sub(" ", s)
    toks = s.split()
    abbr = ABBR_FR if fr else ABBR_EN
    types = STREET_TYPES_FR if fr else STREET_TYPES_EN
    toks = [abbr.get(t, t) for t in toks]
    postal = ""
    plen = 5 if fr or str(country).upper() == "US" else 6
    for t in toks:
        if t.isdigit() and len(t) == plen:
            postal = t
            break
    if not postal and postal_hint is not None and not (isinstance(postal_hint, float) and np.isnan(postal_hint)):
        postal = fold(postal_hint).replace(" ", "")
    # house number: prefer a 1-4 digit token directly before a street type (skipping bis/ter),
    # else the first 1-4 digit token that is not the postal code  (handles "paris 75011 12 r x").
    house, type_i = "", -1
    for i, t in enumerate(toks):
        if t in types:
            type_i = i
            j = i - 1
            while j >= 0 and toks[j] in _ORD:
                j -= 1
            if j >= 0 and _HOUSE.match(toks[j]) and toks[j] != postal:
                house = toks[j]
            break
    if not house:
        for t in toks:
            if _HOUSE.match(t) and t != postal:
                house = t
                break
    if type_i >= 0:
        street = []
        for t in toks[type_i + 1:]:
            if t[0].isdigit() or t in UNIT_WORDS:
                break
            if t not in STOP:
                street.append(t)
            if len(street) >= 5:
                break
        street = [toks[type_i]] + street
        return postal, house, frozenset(street), True
    street = [t for t in toks if not t[0].isdigit() and t not in STOP and t not in UNIT_WORDS]
    return postal, house, frozenset(street), False


def build_keys(rec: pd.DataFrame, id_col: str, name_col: str, addr_col: str, country_col: str,
               postal_col: str | None = None) -> pd.DataFrame:
    """Key table for a set of records (vectorised over python strings; ~1-2M records/min/core)."""
    ids = rec[id_col].to_numpy(dtype=object)
    names = rec[name_col].to_numpy(dtype=object)
    addrs = rec[addr_col].to_numpy(dtype=object)
    ctry = rec[country_col].astype(str).to_numpy(dtype=object)
    posts = rec[postal_col].to_numpy(dtype=object) if postal_col and postal_col in rec.columns else [None] * len(rec)
    nk, nt, nf, pc, hn, st, strong = [], [], [], [], [], [], []
    for n, a, c, p in zip(names, addrs, ctry, posts):
        k, t, f = parse_name(n)
        nk.append(k); nt.append(t); nf.append(f)
        q, h, s, g = parse_addr(a, c, p)
        pc.append(q); hn.append(h); st.append(s); strong.append(g)
    out = pd.DataFrame({"id": ids, "name_key": nk, "name_toks": nt, "name_first": nf, "postal": pc,
                        "house": hn, "street": st, "street_strong": strong, "country": ctry})
    out["building"] = [(f"{c}|{p}|{h}|" + " ".join(sorted(s))) if (h and s) else ""
                       for c, p, h, s in zip(out["country"], out["postal"], out["house"], out["street"])]
    return out


def _jac(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return np.nan
    return len(a & b) / len(a | b)


def name_agree(k1, t1, f1, k2, t2, f2) -> int:
    """2 same key | 1 compatible (jaccard >= .5 or same distinctive first token) | 0 missing | -1 different."""
    if not k1 or not k2:
        return 0
    if k1 == k2:
        return 2
    j = _jac(t1, t2)
    if j >= 0.5 or (f1 == f2 and len(f1) >= 3):
        return 1
    return -1


def _postal_conflict(p1, p2) -> bool:
    """Both present and more than one edit apart (68.75% of true-match postal mismatches are 1-edit typos)."""
    if not p1 or not p2 or p1 == p2:
        return False
    if len(p1) == len(p2):
        return sum(a != b for a, b in zip(p1, p2)) > 1
    return True


def addr_agree(p1, h1, s1, g1, p2, h2, s2, g2) -> int:
    """2 same street + same house | 1 same street, no house conflict | 0 unknown / ambiguous | -1 conflict.
    Street evidence dominates postal evidence: a same-street pair is never a conflict because of the postal
    code alone (typos, departement vs region text, missing codes)."""
    if (not s1 and not h1) or (not s2 and not h2):
        return 0
    house_conf = bool(h1 and h2 and h1 != h2)
    j = _jac(s1, s2)
    if not np.isnan(j) and j >= 0.5:
        if h1 and h1 == h2:
            return 2
        return 0 if house_conf else 1
    if _postal_conflict(p1, p2):
        return -1
    if not np.isnan(j) and j == 0 and g1 and g2:
        return -1
    return 0


def attach_agreement(pairs: pd.DataFrame, s1k: pd.DataFrame, pk: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Compute name_agree / addr_agree for pairs.loc[rows]; others keep 0. s1k keyed by s1_id, pk by cand key."""
    pairs = pairs.copy()
    pairs["name_agree"] = np.int8(0)
    pairs["addr_agree"] = np.int8(0)
    if len(rows) == 0:
        return pairs
    a = s1k.set_index("id")
    b = pk.set_index("id")
    sub = pairs.iloc[rows]
    ia = a.index.get_indexer(sub["s1_id"].to_numpy(dtype=object))
    ib = b.index.get_indexer(sub["cand_key"].to_numpy(dtype=object))
    na = np.zeros(len(sub), np.int8)
    aa = np.zeros(len(sub), np.int8)
    A = {c: a[c].to_numpy(dtype=object) for c in ("name_key", "name_toks", "name_first", "postal", "house", "street", "street_strong")}
    B = {c: b[c].to_numpy(dtype=object) for c in ("name_key", "name_toks", "name_first", "postal", "house", "street", "street_strong")}
    for i, (x, y) in enumerate(zip(ia, ib)):
        if x < 0 or y < 0:
            continue
        na[i] = name_agree(A["name_key"][x], A["name_toks"][x], A["name_first"][x],
                           B["name_key"][y], B["name_toks"][y], B["name_first"][y])
        aa[i] = addr_agree(A["postal"][x], A["house"][x], A["street"][x], A["street_strong"][x],
                           B["postal"][y], B["house"][y], B["street"][y], B["street_strong"][y])
    pairs.iloc[rows, pairs.columns.get_loc("name_agree")] = na
    pairs.iloc[rows, pairs.columns.get_loc("addr_agree")] = aa
    return pairs


def s1_context(s1k: pd.DataFrame) -> pd.DataFrame:
    """Per S1 (label-free, over ALL S1s of the split): how many distinct business names share its building,
    how many S1s share its name key in different places."""
    d = s1k[["id", "name_key", "building", "postal"]].copy()
    b = d[d["building"] != ""]
    crowd = b.groupby("building")["name_key"].nunique()
    d["bldg_n_names"] = d["building"].map(crowd).fillna(1).astype(np.int32)
    n = d[d["name_key"] != ""]
    place = n["building"].where(n["building"] != "", n["postal"])
    tmpl = n.assign(_pl=place).groupby("name_key")["_pl"].nunique()
    d["name_n_places"] = d["name_key"].map(tmpl).fillna(1).astype(np.int32)
    return d.set_index("id")[["bldg_n_names", "name_n_places"]]


# --------------------------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------------------------
DEFAULT_KNOBS = dict(
    crowd_on=True, c1=3,            # multi-tenant building: >= c1 distinct S1 names -> need name_agree >= 1
    tmpl_on=True, c2=3,             # template name: >= c2 places share the S1 name key -> veto on addr conflict
    tmpl_need_addr=False,           # strict variant: template accepts need addr_agree >= 1 (not just no conflict)
    owner_mode="resolve",           # "resolve" | "drop_all" | "off"
    owner_min_claims=2,             # apply the one-owner rule to records claimed by >= this many S1s
    m1=0.15,                        # probability lead needed by the top claimant
    e_gap=2, p_floor=0.5,           # evidence-based winner: evidence lead >= e_gap and p >= p_floor
    rescue_on=False, p_hi=0.9, p_lo=0.3,
)
PRESETS = {
    "default": {},
    "strict": dict(c1=2, c2=2, tmpl_need_addr=True, m1=0.3, e_gap=3, p_floor=0.7),
    "loose": dict(c1=5, tmpl_on=False, m1=0.05, e_gap=1, p_floor=0.4),
    "drop3": dict(crowd_on=False, tmpl_on=False, owner_mode="drop_all", owner_min_claims=3),   # = the 0.975 rule
    "dropall2": dict(crowd_on=False, tmpl_on=False, owner_mode="drop_all", owner_min_claims=2),
    "off": dict(crowd_on=False, tmpl_on=False, owner_mode="off"),
}


def knobs(preset="default", **over) -> dict:
    k = dict(DEFAULT_KNOBS)
    k.update(PRESETS[preset])
    k.update(over)
    return k


# --------------------------------------------------------------------------------------------
# 2-4. Resolve
# --------------------------------------------------------------------------------------------
REASONS = {0: "kept", 1: "veto_crowd", 2: "veto_template", 3: "owner_lost", 4: "owner_dropped_all",
           5: "rescued", 9: "not_accepted"}


def resolve(pairs: pd.DataFrame, k: dict, twins: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """pairs columns: s1_id, cand_key, p, acc (bool, Block 6), name_agree, addr_agree, bldg_n_names,
    name_n_places. Returns (acc_new bool array, reason int8 array) aligned to pairs rows."""
    acc = pairs["acc"].to_numpy(bool).copy()
    reason = np.where(acc, 0, 9).astype(np.int8)
    na = pairs["name_agree"].to_numpy(np.int8)
    aa = pairs["addr_agree"].to_numpy(np.int8)
    p = pairs["p"].to_numpy(np.float64)
    # --- vetoes
    if k.get("crowd_on"):
        v = acc & (pairs["bldg_n_names"].to_numpy() >= k["c1"]) & (na < 1)
        acc[v] = False
        reason[v] = 1
    if k.get("tmpl_on"):
        need = 1 if k.get("tmpl_need_addr") else 0
        v = acc & (pairs["name_n_places"].to_numpy() >= k["c2"]) & (aa < need)
        acc[v] = False
        reason[v] = 2
    # --- one owner per record
    mode = k.get("owner_mode", "resolve")
    if mode != "off":
        idx = np.flatnonzero(acc)
        ck = pairs["cand_key"].to_numpy(dtype=object)[idx]
        codes, uniq = pd.factorize(ck)
        nclaim = np.bincount(codes)
        multi = nclaim[codes] >= k.get("owner_min_claims", 2)
        if multi.any():
            mi = idx[multi]
            mc = codes[multi]
            if mode == "drop_all":
                acc[mi] = False
                reason[mi] = 4
            else:
                ev = na[mi].astype(np.int16) + aa[mi].astype(np.int16)
                pp = p[mi]
                o = np.lexsort((-pp, mc))              # by record, probability desc
                mc_o, pp_o, ev_o, mi_o = mc[o], pp[o], ev[o], mi[o]
                start = np.r_[True, mc_o[1:] != mc_o[:-1]]
                st = np.flatnonzero(start)
                ends = np.r_[st[1:], len(mc_o)]
                for s, e in zip(st, ends):
                    rows = mi_o[s:e]
                    pr, evr = pp_o[s:e], ev_o[s:e]
                    win = -1
                    if pr[0] - pr[1] >= k["m1"] and evr[0] >= evr[1:].max():
                        win = 0
                    else:
                        eo = np.argsort(-evr, kind="stable")
                        if evr[eo[0]] - evr[eo[1]] >= k["e_gap"] and pr[eo[0]] >= k["p_floor"]:
                            win = int(eo[0])
                    for j, r in enumerate(rows):
                        if j != win:
                            acc[r] = False
                            reason[r] = 3 if win >= 0 else 4
    # --- twin rescue
    if k.get("rescue_on") and twins is not None and len(twins):
        acc, reason = _rescue(pairs, acc, reason, p, na, aa, k, twins)
    return acc, reason


def _rescue(pairs, acc, reason, p, na, aa, k, twins):
    ck = pairs["cand_key"].to_numpy(dtype=object)
    s1 = pairs["s1_id"].to_numpy(dtype=object)
    owned = set(ck[acc])
    strong = pd.DataFrame({"s1_id": s1[acc & (p >= k["p_hi"])], "a": ck[acc & (p >= k["p_hi"])]})
    if strong.empty:
        return acc, reason
    tw = pd.concat([twins[["a", "b"]], twins[["b", "a"]].rename(columns={"b": "a", "a": "b"})], ignore_index=True)
    want = strong.merge(tw, on="a")[["s1_id", "b"]].drop_duplicates()
    if want.empty:
        return acc, reason
    cand = np.flatnonzero(~acc & (p >= k["p_lo"]) & (na >= 0) & (aa >= 0))
    if len(cand) == 0:
        return acc, reason
    c = pd.DataFrame({"row": cand, "s1_id": s1[cand], "b": ck[cand]})
    hit = c.merge(want, on=["s1_id", "b"])
    hit = hit[~hit["b"].isin(owned)]
    hit = hit.sort_values("row").drop_duplicates("b")            # a rescued record goes to one S1 only
    acc[hit["row"].to_numpy()] = True
    reason[hit["row"].to_numpy()] = 5
    return acc, reason


# --------------------------------------------------------------------------------------------
# Assembling the input frame
# --------------------------------------------------------------------------------------------
def cand_key(source, cand_id) -> np.ndarray:
    return (pd.Series(source).astype(str).to_numpy(dtype=object) + "|" +
            pd.Series(cand_id).astype(str).to_numpy(dtype=object))


def prepare(pairs: pd.DataFrame, ev, split: str, countries, p_lo_for_keys=0.2,
            name_col="name_latin", addr_col="addr_latin", postal_col="postal_code") -> pd.DataFrame:
    """pairs: s1_id, cand_id, source, country, p, acc (+ label on train). Loads Block 1 records for the
    S1s of `countries` (ALL of them - needed for crowding) and for candidate records that are accepted or
    have p >= p_lo_for_keys, builds keys, and attaches agreement + S1 context columns."""
    t0 = time.time()
    pairs = pairs.reset_index(drop=True).copy()
    pairs["cand_key"] = cand_key(pairs["source"], pairs["cand_id"])
    s1_cols = ["entity_id", "country_norm", name_col, addr_col] + ([postal_col] if postal_col else [])
    s1 = ev.read_records(split, "s1", s1_cols, countries=list(countries))
    s1 = s1.drop_duplicates("entity_id")
    s1k = build_keys(s1, "entity_id", name_col, addr_col, "country_norm", postal_col)
    ctx = s1_context(s1k)
    need = (pairs["acc"].to_numpy(bool)) | (pairs["p"].to_numpy() >= p_lo_for_keys)
    rows = np.flatnonzero(need)
    frames = []
    for src in ("S2", "S3"):
        ids = pairs.loc[need & (pairs["source"].astype(str) == src).to_numpy(), "cand_id"].astype(str).unique()
        if len(ids) == 0:
            continue
        r = ev.read_records(split, src, ["entity_id", "country_norm", name_col, addr_col] +
                            ([postal_col] if postal_col else []), ids=list(ids))
        r["key"] = cand_key([src] * len(r), r["entity_id"])
        frames.append(r.drop_duplicates("key"))
    pool = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["key", "country_norm", name_col, addr_col])
    pk = build_keys(pool, "key", name_col, addr_col, "country_norm", postal_col)
    pairs = attach_agreement(pairs, s1k, pk, rows)
    j = ctx.reindex(pairs["s1_id"].to_numpy(dtype=object))
    pairs["bldg_n_names"] = j["bldg_n_names"].fillna(1).to_numpy(np.int32)
    pairs["name_n_places"] = j["name_n_places"].fillna(1).to_numpy(np.int32)
    print(f"  6.5 prepare {split} {list(countries)}: {len(pairs):,} pairs, keys for {len(s1k):,} S1 / "
          f"{len(pk):,} records, agreement on {len(rows):,} rows ({time.time() - t0:.0f}s)")
    return pairs


def load_twins(path: str, a_col=None, b_col=None, a_src=None, b_src=None) -> pd.DataFrame:
    """Twin edges -> DataFrame(a, b) of cand keys "S2|id". Accepts columns (a_key, b_key) already in key form,
    or (a_source, a_id, b_source, b_id). Pass explicit column names if auto-detection fails."""
    t = pd.read_parquet(path)
    cols = list(t.columns)
    if a_col and b_col:
        if a_src and b_src:
            return pd.DataFrame({"a": cand_key(t[a_src], t[a_col]), "b": cand_key(t[b_src], t[b_col])})
        return pd.DataFrame({"a": t[a_col].astype(str).to_numpy(dtype=object), "b": t[b_col].astype(str).to_numpy(dtype=object)})
    for sa, ia, sb, ib in (("src_a", "id_a", "src_b", "id_b"), ("source_a", "cand_id_a", "source_b", "cand_id_b"),
                           ("a_source", "a_id", "b_source", "b_id")):
        if {sa, ia, sb, ib} <= set(cols):
            return pd.DataFrame({"a": cand_key(t[sa], t[ia]), "b": cand_key(t[sb], t[ib])})
    for ka, kb in (("key_a", "key_b"), ("a", "b"), ("a_key", "b_key"), ("q_key", "nn_key")):
        if {ka, kb} <= set(cols):
            return pd.DataFrame({"a": t[ka].astype(str).to_numpy(dtype=object), "b": t[kb].astype(str).to_numpy(dtype=object)})
    raise ValueError(f"cannot read twin columns {cols}; pass a_col/b_col (and a_src/b_src)")


# --------------------------------------------------------------------------------------------
# Evaluation / tuning (train only)
# --------------------------------------------------------------------------------------------
def active_rows(pairs: pd.DataFrame, p_lo_min=0.2) -> pd.DataFrame:
    """Only rows 6.5 can change: accepted by Block 6, or rescue candidates (p >= p_lo_min). Non-accepted rows
    never count in the metric or the profile, so tuning on this subset is exact and ~10x faster."""
    m = pairs["acc"].to_numpy(bool) | (pairs["p"].to_numpy() >= p_lo_min)
    return pairs[m].reset_index(drop=True)


def per_s1_f(scorer, pairs: pd.DataFrame, acc: np.ndarray) -> np.ndarray:
    from er_eval import f_from_counts
    codes = scorer.codes(pairs["s1_id"].to_numpy())
    lab = pairs["label"].to_numpy().astype(np.int8)
    tp, npred = scorer.counts(codes, acc.astype(bool), lab)
    return f_from_counts(tp, npred, scorer.n_true)


def grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, v)) for v in itertools.product(*[space[k] for k in keys])]


DEFAULT_SPACE = dict(owner_mode=["resolve", "drop_all"], owner_min_claims=[2, 3], m1=[0.05, 0.15, 0.3],
                     crowd_on=[True, False], c1=[3, 5], tmpl_on=[True, False], c2=[3, 5])


def _effective(k: dict) -> tuple:
    """Signature of the knobs that actually change behaviour (drops irrelevant ones)."""
    e = {"owner_mode": k["owner_mode"]}
    if k["owner_mode"] != "off":
        e["owner_min_claims"] = k["owner_min_claims"]
    if k["owner_mode"] == "resolve":
        e.update(m1=k["m1"], e_gap=k["e_gap"], p_floor=k["p_floor"])
    if k["crowd_on"]:
        e["c1"] = k["c1"]
    if k["tmpl_on"]:
        e.update(c2=k["c2"], tmpl_need_addr=k["tmpl_need_addr"])
    if k["rescue_on"]:
        e.update(p_hi=k["p_hi"], p_lo=k["p_lo"])
    return tuple(sorted(e.items()))


def tune(pairs: pd.DataFrame, scorer, twins=None, space=None, base_preset="default", folds=2, seed=0,
         max_configs=400) -> dict:
    """Cross-fit knob search on labelled pairs (CALIBRATION, India/US). Resolution runs on ALL pairs (it is
    coupled across S1s through shared records); only the SCORING is split by S1 fold. Knobs are chosen on
    fold A and reported on fold B (and vice versa), so the reported delta is honest."""
    import er_eval as ev
    pairs = active_rows(pairs)
    space = space or DEFAULT_SPACE
    cfgs = grid(space)
    if len(cfgs) > max_configs:
        rng = np.random.default_rng(seed)
        cfgs = [cfgs[i] for i in rng.choice(len(cfgs), max_configs, replace=False)]
    fold = (ev.s1_hash_unit(scorer.u["s1_id"].to_numpy()) * folds).astype(int)
    f_base = per_s1_f(scorer, pairs, pairs["acc"].to_numpy(bool))
    res = []
    t0 = time.time()
    seen = set()
    for i, c in enumerate(cfgs):
        kk = knobs(base_preset, **c)
        sig = _effective(kk)
        if sig in seen:
            continue                                      # knob combination with identical behaviour
        seen.add(sig)
        a, _ = resolve(pairs, kk, twins)
        f = per_s1_f(scorer, pairs, a)
        res.append((c, f))
        if (i + 1) % 25 == 0:
            print(f"    tune: {i + 1}/{len(cfgs)} configs ({time.time() - t0:.0f}s)")
    cross = np.empty(len(f_base))
    picks = []
    for g in range(folds):
        tr, te = fold != g, fold == g
        best = max(res, key=lambda r: r[1][tr].mean())
        cross[te] = best[1][te]
        picks.append(best[0])
    full_best = max(res, key=lambda r: r[1].mean())
    table = pd.DataFrame([dict(c, s1_macro_f05=f.mean(), delta=f.mean() - f_base.mean()) for c, f in res])
    return {"crossfit_vs_block6": ev.paired_bootstrap(f_base, cross), "fold_picks": picks,
            "best_knobs": full_best[0], "best_full_delta": float(full_best[1].mean() - f_base.mean()),
            "f_block6": f_base, "f_crossfit": cross, "f_best": full_best[1],
            "table": table.sort_values("s1_macro_f05", ascending=False).reset_index(drop=True)}


# --------------------------------------------------------------------------------------------
# 5. Profile (label-free)
# --------------------------------------------------------------------------------------------
def profile(pairs: pd.DataFrame, acc: np.ndarray, s1_universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per country: S1s, share of S1s with no accept, mean accepts per S1, share of accepted records claimed
    by >= 2 S1s (collision), share of accepted pairs in crowded buildings. s1_universe (s1_id, country)
    makes S1s with no candidate rows count as empty too."""
    d = pairs.loc[acc, ["s1_id", "cand_key", "country", "bldg_n_names"]]
    rows = []
    countries = sorted(pairs["country"].astype(str).unique())
    for c in countries:
        dc = d[d["country"].astype(str) == c]
        if s1_universe is not None:
            n_s1 = int((s1_universe["country"].astype(str) == c).sum())
        else:
            n_s1 = int(pairs.loc[pairs["country"].astype(str) == c, "s1_id"].nunique())
        per = dc.groupby("s1_id").size()
        claims = dc.groupby("cand_key").size()
        rows.append({"country": c, "n_s1": n_s1, "empty_share": 1 - len(per) / max(n_s1, 1),
                     "accepts_per_s1": len(dc) / max(n_s1, 1),
                     "accepts_per_nonempty_s1": float(per.mean()) if len(per) else 0.0,
                     "collision_rate": float((claims >= 2).mean()) if len(claims) else 0.0,
                     "records_3plus": int((claims >= 3).sum()),
                     "crowded_accept_share": float((dc["bldg_n_names"] >= 3).mean()) if len(dc) else 0.0})
    return pd.DataFrame(rows)


def reason_counts(pairs, reason) -> pd.DataFrame:
    r = pd.Series(reason).map(REASONS)
    return pd.crosstab(pairs["country"].astype(str).to_numpy(), r.to_numpy())


def france_gate(prof: pd.DataFrame, prof_before: pd.DataFrame, fr="FRANCE", refs=("INDIA", "US"),
                empty_slack=0.05) -> dict:
    """France must move TOWARD India/US on collision rate and accepts per S1, and must not overshoot the
    India/US empty share by more than `empty_slack`."""
    def row(p, c):
        r = p[p["country"] == c]
        return None if r.empty else r.iloc[0]
    a, b = row(prof_before, fr), row(prof, fr)
    ref = prof_before[prof_before["country"].isin(refs)]
    if a is None or b is None or ref.empty:
        return {"pass": None, "why": "missing France or reference rows"}
    tgt = ref[["empty_share", "accepts_per_s1", "collision_rate"]].mean()
    out = {"before": a[["empty_share", "accepts_per_s1", "collision_rate"]].to_dict(),
           "after": b[["empty_share", "accepts_per_s1", "collision_rate"]].to_dict(), "target": tgt.to_dict()}
    ok_coll = abs(b["collision_rate"] - tgt["collision_rate"]) <= abs(a["collision_rate"] - tgt["collision_rate"]) + 1e-12
    ok_acc = abs(b["accepts_per_s1"] - tgt["accepts_per_s1"]) <= abs(a["accepts_per_s1"] - tgt["accepts_per_s1"]) + 1e-12
    ok_empty = b["empty_share"] <= tgt["empty_share"] + empty_slack
    out.update({"collision_toward": bool(ok_coll), "accepts_toward": bool(ok_acc), "empty_ok": bool(ok_empty),
                "pass": bool(ok_coll and ok_acc and ok_empty)})
    return out


# --------------------------------------------------------------------------------------------
# Submission I/O (mirrors whatever format the existing 0.975 file uses)
# --------------------------------------------------------------------------------------------
_SPLIT = re.compile(r"[,;| ]+")


def read_matching(path: str, s1_col=None, match_col=None) -> tuple[pd.DataFrame, dict]:
    """Returns (long frame s1_id, match_id) and a format dict used by write_matching to mirror the input.
    Handles 'one row per S1 with a delimited list' and 'one row per pair'."""
    t = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    cols = list(t.columns)
    s1_col = s1_col or cols[0]
    if match_col is None:
        cand = [c for c in cols if c != s1_col and "match" in c.lower()]
        match_col = cand[0] if cand else [c for c in cols if c != s1_col][0]
    vals = t[match_col].astype(str)
    nonempty = vals[vals.str.len() > 0]
    delim = ","
    for d in (",", ";", "|", " "):
        if nonempty.str.contains(re.escape(d), regex=True).any():
            delim = d
            break
    wrapped = bool(len(nonempty)) and nonempty.str.startswith("[").mean() > 0.5
    long_fmt = t[s1_col].duplicated().any()
    fmt = {"columns": cols, "s1_col": s1_col, "match_col": match_col, "delim": delim, "wrapped": wrapped,
           "long": bool(long_fmt), "n_rows": len(t)}
    if long_fmt:
        out = t[[s1_col, match_col]].rename(columns={s1_col: "s1_id", match_col: "match_id"})
        out = out[out["match_id"].str.len() > 0]
    else:
        s = vals.str.strip("[]").str.replace("'", "").str.replace('"', "")
        lst = s.apply(lambda x: [y for y in _SPLIT.split(x) if y] if x else [])
        out = pd.DataFrame({"s1_id": t[s1_col].repeat(lst.str.len()).to_numpy(),
                            "match_id": np.concatenate(lst.to_numpy()) if lst.str.len().sum() else np.array([], object)})
    fmt["all_s1"] = t[s1_col].tolist()
    return out.reset_index(drop=True), fmt


def write_matching(long: pd.DataFrame, fmt: dict, path: str):
    s1c, mc = fmt["s1_col"], fmt["match_col"]
    if fmt["long"]:
        out = long.rename(columns={"s1_id": s1c, "match_id": mc})[[s1c, mc]]
    else:
        g = long.groupby("s1_id")["match_id"].apply(lambda x: fmt["delim"].join(sorted(set(x))))
        out = pd.DataFrame({s1c: fmt["all_s1"]})
        out[mc] = out[s1c].map(g).fillna("")
        if fmt["wrapped"]:
            out[mc] = "[" + out[mc] + "]"
        extra = [c for c in fmt["columns"] if c not in (s1c, mc)]
        if extra:
            print(f"  WARNING: base file had extra columns {extra}; they are written empty")
            for c in extra:
                out[c] = ""
        out = out[fmt["columns"]]
    tmp = path + ".tmp"
    out.to_csv(tmp, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    os.replace(tmp, path)
    return out


def splice(base_long: pd.DataFrame, new_long: pd.DataFrame, s1_country: pd.Series, replace_countries) -> pd.DataFrame:
    """Rows of S1s in replace_countries come from new_long, all others from base_long (one country group
    changed per upload -> the leaderboard delta measures exactly that group)."""
    ctry = base_long["s1_id"].map(s1_country).astype(str)
    keep = base_long[~ctry.isin(replace_countries).to_numpy()]
    nc = new_long["s1_id"].map(s1_country).astype(str)
    add = new_long[nc.isin(replace_countries).to_numpy()]
    return pd.concat([keep, add], ignore_index=True)


def validate(long: pd.DataFrame, cand_pairs: pd.DataFrame | None, fmt: dict, s1_country: pd.Series) -> dict:
    """cand_pairs: long (s1_id, match_id) view of candidate_pairs.tsv (optional but recommended)."""
    rep = {"n_pairs": int(len(long)), "dup_pairs": int(long.duplicated().sum()),
           "unknown_s1": int((~long["s1_id"].isin(set(fmt["all_s1"]))).sum())}
    claims = long.groupby("match_id")["s1_id"].nunique()
    rep["records_claimed_2plus"] = int((claims >= 2).sum())
    rep["records_claimed_3plus"] = int((claims >= 3).sum())
    if cand_pairs is not None:
        m = long.merge(cand_pairs.drop_duplicates(), on=["s1_id", "match_id"], how="left", indicator=True)
        rep["not_in_candidates"] = int((m["_merge"] == "left_only").sum())
    ctry = long["s1_id"].map(s1_country).astype(str)
    rep["pairs_by_country"] = ctry.value_counts().to_dict()
    rep["ok"] = rep["dup_pairs"] == 0 and rep["unknown_s1"] == 0 and rep.get("not_in_candidates", 0) == 0
    return rep


def diff_vs(base_long: pd.DataFrame, new_long: pd.DataFrame, s1_country: pd.Series) -> pd.DataFrame:
    a = base_long.groupby("s1_id")["match_id"].apply(frozenset)
    b = new_long.groupby("s1_id")["match_id"].apply(frozenset)
    ids = a.index.union(b.index)
    a, b = a.reindex(ids), b.reindex(ids)
    changed = pd.Series([x != y for x, y in zip(a, b)], index=ids)
    c = pd.Series(ids, index=ids).map(s1_country).astype(str)
    out = pd.DataFrame({"changed": changed, "country": c})
    return out.groupby("country")["changed"].agg(["sum", "mean"]).rename(columns={"sum": "s1_changed",
                                                                                   "mean": "share_of_s1_with_output"})


def log_upload(path: str, entry: dict):
    """Pre-register an upload (parent, change, expected direction) BEFORE submitting it."""
    log = json.load(open(path)) if os.path.exists(path) else []
    entry = dict(entry, time=time.strftime("%F %T"), code_version=B6_CODE_VERSION)
    log.append(entry)
    json.dump(log, open(path, "w"), indent=1, default=str)
    return log


# --------------------------------------------------------------------------------------------
# Loading pair probabilities + Block 2 context
# --------------------------------------------------------------------------------------------
PROB_COLS = ("p", "prob", "p_blend", "p_ens", "match_prob", "prob_v1", "p_v1", "proba", "score_b5")


def load_prob_glob(pattern: str, prob_col: str | None = None) -> pd.DataFrame:
    """Every parquet matching `pattern` (glob, recursive **) -> s1_id, cand_id, p. Auto-detects the
    probability column unless prob_col is given."""
    import glob as _g
    files = sorted(_g.glob(pattern, recursive=True))
    files = [f for f in files if not f.endswith(".tmp")]
    if not files:
        raise FileNotFoundError(pattern)
    import pyarrow.parquet as pq
    cols0 = pq.read_schema(files[0]).names
    pc = prob_col or next((c for c in PROB_COLS if c in cols0), None)
    if pc is None:
        raise ValueError(f"no probability column in {files[0]} (columns {cols0}); pass prob_col=")
    fr = [pq.read_table(f, columns=["s1_id", "cand_id", pc]).to_pandas() for f in files]
    out = pd.concat(fr, ignore_index=True).rename(columns={pc: "p"})
    out = out.drop_duplicates(["s1_id", "cand_id"])
    print(f"  probs: {len(out):,} rows from {len(files)} files (column '{pc}')")
    return out


def attach_b2(probs: pd.DataFrame, b2p, split, roles, countries, k) -> pd.DataFrame:
    """Block 2 rows (source, country, rank, role, label) for split/roles/countries, left-joined with p.
    Rows without a probability get p = 0 (never accepted)."""
    fr = []
    for _m, df in b2p.iter_candidates(split, roles=roles, countries=countries, k=k):
        if len(df):
            fr.append(df[[c for c in ("s1_id", "cand_id", "source", "country", "rank", "role", "label") if c in df.columns]])
    base = pd.concat(fr, ignore_index=True)
    out = base.merge(probs, on=["s1_id", "cand_id"], how="left")
    miss = out["p"].isna().mean()
    if miss > 0.01:
        print(f"  WARNING: {miss:.2%} of Block 2 rows have no probability (set to 0)")
    out["p"] = out["p"].fillna(0.0).astype(np.float64)
    return out


def accept_from_long(pairs: pd.DataFrame, long: pd.DataFrame) -> np.ndarray:
    """acc = (s1_id, cand_id) present in a submission's long frame (s1_id, match_id)."""
    k = pd.MultiIndex.from_frame(long[["s1_id", "match_id"]].astype(str))
    q = pd.MultiIndex.from_arrays([pairs["s1_id"].astype(str), pairs["cand_id"].astype(str)])
    acc = q.isin(k)
    missing = len(long) - int(acc.sum())
    if missing > 0:
        print(f"  WARNING: {missing:,} accepted pairs of the file are not among the loaded pairs")
    return np.asarray(acc)


def pairs_to_long(pairs: pd.DataFrame, acc: np.ndarray) -> pd.DataFrame:
    return pairs.loc[acc, ["s1_id", "cand_id"]].rename(columns={"cand_id": "match_id"}).astype(str).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Label-free preset ranking for FRANCE (distance of France's output profile to India/US's)
# --------------------------------------------------------------------------------------------
CANDIDATE_PRESETS = [("default", {}), ("loose", {}), ("default", {"tmpl_on": False}), ("strict", {"tmpl_on": False}),
                     ("strict", {}), ("dropall2", {"crowd_on": True, "tmpl_on": True})]


def profile_distance(g: dict) -> float:
    t, a = pd.Series(g["target"]), pd.Series(g["after"])
    return float(((a - t).abs() / t.abs().clip(lower=1e-3)).sum())


def rank_presets(pairs, universe, before, twins=None, candidates=None, extra=None) -> pd.DataFrame:
    """Resolve with every candidate preset; return France gate + profile distance, best (smallest) first.
    On the synthetic fixture the distance ordered presets almost exactly like the true France F0.5
    (it is a proxy, not a label: the leaderboard decides)."""
    pairs = active_rows(pairs)
    rows = []
    for name, over in (candidates or CANDIDATE_PRESETS):
        k = knobs(name, **over) | (extra or {})
        a, _ = resolve(pairs, k, twins)
        g = france_gate(profile(pairs, a, universe), before)
        if g.get("pass") is None:
            continue
        rows.append({"preset": name, "over": json.dumps(over), "knobs": k, "gate": g["pass"],
                     "dist": profile_distance(g), **{f"fr_{kk}": v for kk, v in g["after"].items()}})
    t = pd.DataFrame(rows)
    return t.sort_values(["gate", "dist"], ascending=[False, True]).reset_index(drop=True)


def save_upload(up_dir: str, name: str, long: pd.DataFrame, fmt: dict, cand_long, s1_country, entry: dict,
                overwrite: bool = False) -> dict:
    """Validate, then write <up_dir>/<name>/matching_results.tsv ONCE and pre-register it in ladder_log.json.
    An existing upload is never overwritten (it may already be on the leaderboard) unless overwrite=True."""
    path = os.path.join(up_dir, name, "matching_results.tsv")
    if os.path.exists(path) and not overwrite:
        print(f"  {name} already exists - kept as is ({path}); delete the folder or pass overwrite=True to rebuild")
        return {"ok": True, "kept_existing": True}
    rep = validate(long, cand_long, fmt, s1_country)
    print(f"  {name} validation:", rep)
    if not rep["ok"]:
        raise AssertionError(f"{name} failed validation - not written")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_matching(long, fmt, path)
    log_upload(os.path.join(up_dir, "ladder_log.json"), dict(entry, upload=name, validation=rep))
    print(f"  {name} written: {path}  (candidate_pairs.tsv: reuse the 0.975 one, unchanged)")
    return rep
