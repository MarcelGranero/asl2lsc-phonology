#!/usr/bin/env python3
"""ASL-LEX 2.0 sub-handshape prediction -> LSC handshape (SignHub) total mapping.

Enumerates every combination of the ASL-LEX 2.0 value sets and resolves each to:
  ok        -> exactly one LSC config (+ benchmark label)
  ambiguous -> >1 LSC configs share the tuple (C1 6-family / C2,C3 7 vs index+thumb)
  error     -> typed reason; Crossed/Stacked map to no LSC configuration,
               Bent bridges to curved.

Matching rules (table NA = wildcard; predicted NA compatible with anything):
  lsc_selected_fingers = 't' if SF=='thumb' else ('t' if ThumbPosition=='Open' else '')+SF
  lsc_flexion          = FullyOpen->extended, Flat->flat, Curved->curved,
                         Bent->curved, FullyClosed->closed
  spread, thumb_contact: equal or either side NA
"""
import json, csv, itertools
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent
CONFIGS = json.load(open(HERE / "signhub_asllex_map.json"))["configs"]

SF_VALUES = ["i","m","r","p","im","ir","ip","mr","mp","rp",
             "imr","imp","irp","mrp","imrp","thumb"]
SF_UNATTESTED = {"ir","mp","rp"}          # unattested in ASL per ASL-LEX
FLEXION_VALUES = ["FullyOpen","Flat","Curved","Bent","FullyClosed","Crossed","Stacked","NA"]
SPREAD_VALUES = ["1","0","NA"]
THUMBPOS_VALUES = ["Open","Closed"]
CONTACT_VALUES = ["0","1","NA"]

# Bent -> curved: LSC has no 'bent' flexion; curved is the nearest LSC
# configuration, so Bent predictions resolve instead of erroring out.
FLEXION_BRIDGE = {"FullyOpen":"extended","Flat":"flat","Curved":"curved",
                  "FullyClosed":"closed","Bent":"curved"}

def _compat(pred, table):
    return pred == "NA" or table == "NA" or pred == table

def map_prediction(sf, flexion, spread, thumbpos, contact):
    """-> dict(status='ok'|'ambiguous'|'error', ...)"""
    # The PopSign feature vocabulary spells thumb-only selected fingers 't'
    # (results/feature_vocab.json), while this table and SF_VALUES use
    # 'thumb'. Every branch below tests `sf == "thumb"`, so an unnormalised
    # 't' would fall through all of them and resolve as
    # no_LSC_config_for_combination -- every thumb-only handshape scored
    # as unmappable.
    if sf == "t":
        sf = "thumb"
    if flexion == "Crossed":
        return {"status":"error","reason":"crossed_flexion_not_in_LSC"}
    if flexion == "Stacked":
        return {"status":"error","reason":"stacked_flexion_not_in_LSC"}
    if sf == "thumb" and thumbpos == "Closed":
        return {"status":"error","reason":"inconsistent_selected_thumb_implies_Open"}
    if flexion == "NA" and sf != "thumb":
        return {"status":"error","reason":"inconsistent_flexion_NA_needs_thumb_only"}

    lsc_sf = "t" if sf == "thumb" else (("t" if thumbpos == "Open" else "") + sf)
    lsc_flex = FLEXION_BRIDGE.get(flexion)   # None when flexion == 'NA' (thumb-only)

    hits = [c for c in CONFIGS
            if c["lsc_selected_fingers"] == lsc_sf
            and (lsc_flex is None or c["lsc_flexion"] == lsc_flex)
            and _compat(spread, c["asllex_Spread"])
            and _compat(contact, c["asllex_ThumbContact"])]
    if not hits:
        return {"status":"error","reason":"no_LSC_config_for_combination"}
    if len(hits) == 1:
        c = hits[0]
        return {"status":"ok","lsc_config":c["signhub_name"],
                "benchmark_label":c["benchmark_label"]}
    return {"status":"ambiguous",
            "candidates":"|".join(c["signhub_name"] for c in hits)}

def main():
    rows, stats, per_config = [], Counter(), Counter()
    for sf, fx, sp, tp, tc in itertools.product(
            SF_VALUES, FLEXION_VALUES, SPREAD_VALUES, THUMBPOS_VALUES, CONTACT_VALUES):
        r = map_prediction(sf, fx, sp, tp, tc)
        out = {"SelectedFingers":sf,"Flexion":fx,"Spread":sp,
               "ThumbPosition":tp,"ThumbContact":tc,
               "status":r["status"],
               "lsc_config":r.get("lsc_config",""),
               "benchmark_label":r.get("benchmark_label",""),
               "candidates":r.get("candidates",""),
               "reason":r.get("reason",""),
               "note":"SF unattested in ASL-LEX" if sf in SF_UNATTESTED else ""}
        rows.append(out)
        key = r["status"] if r["status"] != "error" else f"error:{r['reason']}"
        stats[key] += 1
        if r["status"] == "ok":
            per_config[r["lsc_config"]] += 1

    with open(HERE/"asllex_to_lsc_lookup.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"total combinations: {len(rows)}")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<48} {v}")
    unreached = [c["signhub_name"] for c in CONFIGS if per_config[c["signhub_name"]] == 0]
    print(f"\nconfigs reachable as unique 'ok': {sum(1 for c in CONFIGS if per_config[c['signhub_name']]>0)}/43")
    print("unreachable (only via ambiguous or never):", unreached)

if __name__ == "__main__":
    main()
