"""Brand registry for the global panel-forecasting benchmark."""
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(_THIS_DIR, "data", "people counting data", "PEOPLE COUNTING DATA")


def region_koton(store_name: str) -> str:
    n = store_name.upper()
    prefixes = {
        "TUR_": "Turkey", "ALB_": "Albania", "BOS_": "Bosnia", "GEO_": "Georgia",
        "EGY_": "Egypt", "HUN_": "Hungary", "MKD_": "Macedonia",
    }
    for p, region in prefixes.items():
        if n.startswith(p) or f"_{p}" in n:
            return region
    if n.startswith("BF RU") or "_RU_" in n or n.startswith("RU "):
        return "Russia"
    if "HUN" in n and "M1483" in n:
        return "Hungary"
    if "MACEDONIA" in n:
        return "Macedonia"
    return "Turkey"


NON_CHINA_AWANG_STORES = {
    "American Dream": "USA", "American Dream Fitting Room": "USA",
    "Las Vegas Fontainebleau": "USA", "Miami": "USA", "Miami Pop Up": "USA",
    "Soho NY": "USA", "South Coast Plaza": "USA", "Toronto": "Canada",
    "Hong Kong Harbour City": "Hong Kong", "Macau Londoner": "Macau",
    "Taiwan Breeze": "Taiwan",
}


def region_awang(store_name: str) -> str:
    return NON_CHINA_AWANG_STORES.get(store_name, "China")


def region_fixed(region_name):
    def _fn(store_name: str) -> str:
        return region_name
    return _fn


# brand_key -> dict(csv, region_fn, min_count (floor for anomaly abs threshold))
BRANDS = {
    "koton": dict(csv="koton.csv", region_fn=region_koton),
    "alexander_wang": dict(csv="alexander wang.csv", region_fn=region_awang),
    "vatan": dict(csv="vatan.csv", region_fn=region_fixed("Turkey")),
    "lcw": dict(csv="LCW.csv", region_fn=region_fixed("Turkey")),
    "mad_parfume": dict(csv="mad_parfume.csv", region_fn=region_fixed("Turkey")),
    "bluemint": dict(csv="Bluemint.csv", region_fn=region_fixed("Turkey")),
    "opap": dict(csv="OPAP.csv", region_fn=region_fixed("Greece")),
    "tai_loy": dict(csv="tai_loy.csv", region_fn=region_fixed("Peru")),
}

RAW_COLUMNS = ["col1", "store_id", "store_name", "count_in", "count_out", "ts"]
