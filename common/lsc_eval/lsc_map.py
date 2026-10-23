"""ASL-LEX phonological attributes -> LSC handshape: canonicalisation,
the 5-channel distance, and full inventory ranking.

An ASL-trained model predicts five attribute streams -- selected fingers,
flexion, spread, thumb position, thumb contact -- with independent heads, so
nothing stops it emitting a combination that is not a real LSC handshape.
This module ranks every valid LSC configuration by distance from the
predicted attribute vector; the nearest one is the predicted handshape.

Distance (the benchmark's ASL-LSC mapping definition) is the sum over five
channels:

  selected fingers   symmetric difference of the finger sets (adding or
                     deleting a finger costs 1; substitution is delete+add=2)
  flexion            shortest path on the 4-node graph FullyOpen / Flat /
                     Curved-Bent / FullyClosed; only FullyOpen<->FullyClosed
                     is 2 hops, every other distinct pair is 1
  spread             1 for any change, NA included
  thumb position     1 for any change, NA included
  thumb contact      1 for any change, NA included

ASL-LEX distinguishes Curved from Bent; LSC does not, so both collapse onto
one flexion node before any distance is computed.  Flexion NA / Stacked /
Crossed and finger sets outside 'imrpt' have no LSC counterpart at all: they
are rejected (ValueError) rather than guessed at, and the evaluator counts
such predictions as unmappable.

This is a self-contained re-implementation of the benchmark's
labels/lsc_handshape_distance.py (which the per-leaf evaluators import for
constrained decoding).  Its deterministic tie order is preserved exactly:
equal-distance feature sets are ordered by (selected-finger edits, first
alphabetical name).

Stdlib only; runs under any python >= 3.6.
"""

import csv

STREAMS = ("selected_fingers", "flexion", "spread", "thumb_position",
           "thumb_contact")

# --------------------------------------------------------------------------
# Loose canonicalisation -- the shared string space in which predicted and
# gold ATTRIBUTE values are compared.  Model vocabularies and the benchmark
# labels spell the same values differently ("Fully Open" vs "FullyOpen",
# "1.0" vs "1", "__NA__" vs "NA"); ids overlap numerically without agreeing,
# so everything is matched through canonical strings, never raw ints.
# Values with no LSC counterpart (Stacked, Crossed, flexion NA) survive this
# layer unchanged -- they are legitimate *attribute* predictions that simply
# score as wrong; only the strict layer below rejects them.
# --------------------------------------------------------------------------

_NA_ISH = {"__na__", "", "na", "n/a", "nan", "none"}

_FLEX_LOOSE = {
    "fully open": "FullyOpen", "fullyopen": "FullyOpen",
    "fully closed": "FullyClosed", "fullyclosed": "FullyClosed",
    "curved": "Curved/Bent", "bent": "Curved/Bent",
    "curved/bent": "Curved/Bent", "flat": "Flat",
    "stacked": "Stacked", "crossed": "Crossed",
}


def canon_loose(stream, value):
    """One canonical spelling per value, applied to BOTH sides."""
    s = str(value).strip()
    if s.lower() in _NA_ISH:
        return "NA"
    if stream == "flexion":
        return _FLEX_LOOSE.get(s.lower(), s)
    if stream in ("spread", "thumb_contact"):
        try:
            return str(int(float(s)))  # "1.0" / "1" -> "1"
        except ValueError:
            return s
    return s  # selected_fingers, thumb_position are already identical


# --------------------------------------------------------------------------
# Strict canonicalisation -- the snapping space.  Applied on top of the
# loose layer; raises ValueError for values no LSC handshape can have.
# --------------------------------------------------------------------------

VALID_FINGERS = frozenset("imrpt")

_FLEX_STRICT = {
    "fullyopen": "FullyOpen", "open": "FullyOpen", "extended": "FullyOpen",
    "flat": "Flat",
    "curved": "Curved/Bent", "bent": "Curved/Bent",
    "curved/bent": "Curved/Bent", "curvedbent": "Curved/Bent",
    "fullyclosed": "FullyClosed", "closed": "FullyClosed",
}
_SPREAD_STRICT = {"0": "0", "1": "1", "na": "NA", "n/a": "NA", "": "NA",
                  "none": "NA", "notspread": "0", "spread": "1"}
_CONTACT_STRICT = {"0": "0", "1": "1", "na": "NA", "n/a": "NA", "": "NA",
                   "none": "NA", "nocontact": "0", "contact": "1"}
_POSITION_STRICT = {"open": "Open", "closed": "Closed",
                    "na": "NA", "n/a": "NA", "": "NA", "none": "NA"}


def _strict(value, table, channel):
    key = str(value).strip().lower().replace(" ", "").replace("_", "")
    try:
        return table[key]
    except KeyError:
        raise ValueError(
            "unrecognised %s value %r; expected one of %s"
            % (channel, value, sorted(set(table.values()))))


def canon_fingers(value):
    """Canonicalise a selected-fingers string to sorted i<m<r<p<t order."""
    chars = [c for c in str(value).strip().lower() if c not in "-_ /+"]
    if not chars:
        raise ValueError("selected_fingers is empty; expected some of 'imrpt'")
    bad = sorted(set(chars) - VALID_FINGERS)
    if bad:
        raise ValueError(
            "unrecognised finger character(s) %s in selected_fingers %r"
            % (bad, value))
    if len(set(chars)) != len(chars):
        raise ValueError("repeated finger character in selected_fingers %r"
                         % (value,))
    return "".join(sorted(set(chars), key="imrpt".index))


def strict_features(selected_fingers, flexion, spread, thumb_position,
                    thumb_contact):
    """Five attribute values -> canonical feature tuple, or ValueError."""
    return (
        canon_fingers(selected_fingers),
        _strict(flexion, _FLEX_STRICT, "flexion"),
        _strict(spread, _SPREAD_STRICT, "spread"),
        _strict(thumb_position, _POSITION_STRICT, "thumb_position"),
        _strict(thumb_contact, _CONTACT_STRICT, "thumb_contact"),
    )


# --------------------------------------------------------------------------
# Distance
# --------------------------------------------------------------------------

def fingers_distance(a, b):
    return len(set(a) ^ set(b))


def flexion_distance(a, b):
    if a == b:
        return 0
    return 2 if {a, b} == {"FullyOpen", "FullyClosed"} else 1


def distance(q, f):
    """Total sheet distance between two canonical feature tuples."""
    return (fingers_distance(q[0], f[0]) + flexion_distance(q[1], f[1])
            + (q[2] != f[2]) + (q[3] != f[3]) + (q[4] != f[4]))


# --------------------------------------------------------------------------
# LSC inventory
# --------------------------------------------------------------------------

REQUIRED_COLUMNS = (
    "Handshape Name",
    "ASL-LEX Selected Fingers",
    "ASL-LEX Flexion",
    "ASL-LEX Spread",
    "ASL-LEX Thumb Position",
    "ASL-LEX Thumb Contact",
)


def load_inventory(csv_path):
    """Read the valid LSC handshapes as [(feature tuple, [names])].

    Distinct handshape names with identical ASL-LEX features (the sheet's
    "[merged]" pairs) are grouped onto one feature set; names are sorted.
    Only the ASL-LEX-coded columns are used: both sides of the distance are
    ASL-LEX coded.

    Lines whose first non-blank character is '#' are comments and skipped
    (inventory/lsc_handshapes.csv keeps the six benchmark-unavailable
    handshapes as commented-out rows).  No data field contains '#', and the
    utf-8-sig codec strips any BOM before the check.
    """
    with open(str(csv_path), newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(
            line for line in fh if not line.lstrip().startswith("#"))
        fields = {f.strip(): f for f in (reader.fieldnames or [])}
        missing = [c for c in REQUIRED_COLUMNS if c not in fields]
        if missing:
            raise ValueError("%s: missing column(s) %s" % (csv_path, missing))
        groups, seen = {}, {}
        for r in reader:
            name = (r.get(fields["Handshape Name"]) or "").strip()
            if not name:
                continue  # blank spacer rows are common in sheet exports
            feats = strict_features(
                r[fields["ASL-LEX Selected Fingers"]],
                r[fields["ASL-LEX Flexion"]],
                r[fields["ASL-LEX Spread"]],
                r[fields["ASL-LEX Thumb Position"]],
                r[fields["ASL-LEX Thumb Contact"]])
            if name in seen and seen[name] != feats:
                raise ValueError(
                    "%s: handshape %r appears twice with different features"
                    % (csv_path, name))
            seen[name] = feats
            names = groups.setdefault(feats, [])
            if name not in names:
                names.append(name)
    if not groups:
        raise ValueError("%s: no handshape rows found" % (csv_path,))
    return sorted(((feats, sorted(names)) for feats, names in groups.items()),
                  key=lambda x: x[1][0])


def norm_name(s):
    """Handshape-name comparison key: case- and _/- insensitive."""
    return str(s).strip().lower().replace("_", "-")


# --------------------------------------------------------------------------
# Ranking and snapping
# --------------------------------------------------------------------------

def rank(query, inventory):
    """All inventory feature sets ordered by distance from `query`.

    Returns [(total distance, feature tuple, [names])].  Ties are broken
    deterministically by (selected-finger edits, first alphabetical name),
    matching the tie order of the original shared mapping module so the
    strict top-1 prediction is reproducible.
    """
    scored = [(distance(query, feats), fingers_distance(query[0], feats[0]),
               feats, names) for feats, names in inventory]
    scored.sort(key=lambda s: (s[0], s[1], s[3][0]))
    return [(d, feats, names) for d, _, feats, names in scored]


def snap(query, inventory):
    """Nearest valid LSC handshape(s) for a canonical feature tuple.

    Returns a dict:
      name        deterministic single predicted handshape (see rank())
      candidates  every name at the minimum distance (ties are real and are
                  always returned in full rather than silently broken)
      distance    the minimum distance (0 = query is already valid LSC)
      is_exact / is_tie
      ranked      the full ranking, for top-k metrics
    """
    ranked = rank(query, inventory)
    best = ranked[0][0]
    winners = [r for r in ranked if r[0] == best]
    candidates = sorted(set(n for _, _, names in winners for n in names))
    return {
        "name": winners[0][2][0],
        "candidates": candidates,
        "distance": best,
        "is_exact": best == 0,
        "is_tie": len(winners) > 1,
        "ranked": ranked,
    }


def expected_recall_at(ranked, gold, k):
    """Probability gold is in a top-k list built by uniform tie-breaking.

    Generalises the k=1 expected accuracy to retrieval.  Names are ordered by
    distance; within a tied block the order is arbitrary, so we model it as a
    uniform random permutation and ask P(gold lands in the first k names).

    With `before` = names strictly closer than gold, `tied` = names at gold's
    distance (gold included), and room = k - before:

        room <= 0          -> 0.0   (the tie block starts at or past k)
        room >= tied       -> 1.0   (the whole block fits inside k)
        otherwise          -> room / tied

    The middle case is the straddling tie: k cuts the block, and a uniform
    permutation puts gold in the visible part with probability room/tied.  At
    k=1 this reduces to 1/|tied| when gold ties for nearest, matching
    `evaluate.py`'s expected_credit exactly.  Returns None if `gold` names no
    inventory entry.
    """
    g = norm_name(gold)
    gold_dist = None
    for d, _, names in ranked:
        if any(norm_name(n) == g for n in names):
            gold_dist = d if gold_dist is None else min(gold_dist, d)
    if gold_dist is None:
        return None
    # Count NAMES, not feature sets: the two conflated pairs are two distinct
    # identities a guesser must choose between.
    before = sum(len(names) for d, _, names in ranked if d < gold_dist)
    tied = sum(len(names) for d, _, names in ranked if d == gold_dist)
    room = k - before
    if room <= 0:
        return 0.0
    if room >= tied:
        return 1.0
    return room / float(tied)


def gold_rank(ranked, gold):
    """Retrieval rank of the gold handshape in a ranking.

    rank = 1 + number of non-gold feature sets STRICTLY closer than the
    gold's feature set, so ties share the best rank: gold is retrieved at k
    iff gold_rank <= k, and gold_rank == 1 iff gold is among the tie-any
    candidates.  Returns None if `gold` names no inventory entry.
    """
    g = norm_name(gold)
    gold_dist = None
    for d, _, names in ranked:
        if any(norm_name(n) == g for n in names):
            gold_dist = d if gold_dist is None else min(gold_dist, d)
    if gold_dist is None:
        return None
    closer = sum(1 for d, _, names in ranked
                 if d < gold_dist and not any(norm_name(n) == g for n in names))
    return closer + 1
