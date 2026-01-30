import json
from pathlib import Path

import pandas as pd


IN_JSON = Path("processed") / "cleaning_summary.json"
OUT_CSV = Path("processed") / "cleaning_key_stats.csv"


def main() -> None:
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))
    rows = []
    for m in data:
        vc = m.get("value_counts", {})
        nt = vc.get("network_type_top", {})
        rows.append(
            {
                "file": m.get("file"),
                "dt_p05_s": m.get("dt_p05_s"),
                "dt_median_s": m.get("dt_median_s"),
                "dt_p95_s": m.get("dt_p95_s"),
                "network_type_1": list(nt.keys())[0] if len(nt) >= 1 else None,
                "network_type_1_count": list(nt.values())[0] if len(nt) >= 1 else None,
                "network_type_2": list(nt.keys())[1] if len(nt) >= 2 else None,
                "network_type_2_count": list(nt.values())[1] if len(nt) >= 2 else None,
                "network_type_3": list(nt.keys())[2] if len(nt) >= 3 else None,
                "network_type_3_count": list(nt.values())[2] if len(nt) >= 3 else None,
                "screen_on_false": vc.get("screen_on", {}).get("False"),
                "screen_on_true": vc.get("screen_on", {}).get("True"),
                "is_charging_false": vc.get("is_charging", {}).get("False"),
                "is_charging_true": vc.get("is_charging", {}).get("True"),
                "logger_package_top": list(vc.get("logger_package", {}).keys())[0] if vc.get("logger_package") else None,
            }
        )

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    main()

