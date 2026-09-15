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


# Swatch is genuinely global (~194 stores across ~25 countries), and unlike
# the other brands there's no clean prefix code in the store names -- just
# city names, often after a comma (e.g. "Akasya, Istanbul"), sometimes bare
# (e.g. "Bordeaux" for France). This is a best-effort keyword match built by
# reading the actual store list, not verified ground truth -- a handful of
# genuinely ambiguous names (shared mall names across countries, unfamiliar
# ones) fall through to "Other", which still gets the generic global
# holidays (New Year/Valentine's/Mother's Day/Black Friday) from holidays.py,
# just not country-specific ones.
_SWATCH_KEYWORDS = [
    # Turkey
    ("Istanbul", "Turkey"), ("Ankamall", "Turkey"), ("Ankara", "Turkey"),
    ("Antalya", "Turkey"), ("Terracity", "Turkey"), ("Akasya", "Turkey"),
    ("Buyaka", "Turkey"), ("Ozdilek", "Turkey"), ("Istinye", "Turkey"),
    ("Palladium Mall", "Turkey"),
    # Switzerland
    ("Zurich", "Switzerland"), ("Zermatt", "Switzerland"), ("Biel", "Switzerland"),
    ("Basel", "Switzerland"), ("Geneva", "Switzerland"), ("Wallisellen", "Switzerland"),
    ("Interlaken", "Switzerland"), ("Lausanne", "Switzerland"), ("Bern", "Switzerland"),
    ("Crans Montana", "Switzerland"), ("Lugano", "Switzerland"), ("Luzern", "Switzerland"),
    ("Mendrisio", "Switzerland"), ("Aubonne", "Switzerland"), ("Verbier", "Switzerland"),
    # Austria
    ("Kaerntnerstrasse", "Austria"), ("Graben", "Austria"), ("Kaufhaus Tyrol", "Austria"),
    ("Vösendorf", "Austria"), ("Linz", "Austria"), ("Sporgasse", "Austria"),
    ("Getreidegasse", "Austria"),
    # France
    ("Aix", "France"), ("Bordeaux", "France"), ("CAP 3000", "France"), ("Cannes", "France"),
    ("Sénart", "France"), ("Deauville", "France"), ("Dijon", "France"), ("Lille", "France"),
    ("Lyon", "France"), ("Marseille", "France"), ("Nantes", "France"), ("Paris", "France"),
    ("Parly", "France"), ("Montpellier", "France"), ("Rosny", "France"), ("Rouen", "France"),
    ("Strasbourg", "France"), ("Toulouse", "France"), ("Val d'Europe", "France"),
    ("Rennes", "France"), ("Nice", "France"),
    # Poland
    ("Warszawa", "Poland"), ("Warsaw", "Poland"), ("Kraków", "Poland"), ("Gdansk", "Poland"),
    ("Lodz", "Poland"), ("Mokotow", "Poland"), ("Mlociny", "Poland"), ("Zlote", "Poland"),
    ("Poznan", "Poland"), ("Wroclavia", "Poland"),
    # UK
    ("London", "UK"), ("Edinburgh", "UK"), ("Glasgow", "UK"), ("Sheffield", "UK"),
    ("Covent Garden", "UK"), ("Paradise Street", "UK"), ("Newcastle Metrocentre", "UK"),
    ("Trafford Centre", "UK"), ("Vivacity", "UK"), ("White City", "UK"),
    # Australia
    ("Bondi", "Australia"), ("Collins Street", "Australia"), ("Murray St", "Australia"),
    ("Pitt Street", "Australia"), ("Queen Street", "Australia"),
    # Malaysia
    ("Kota Bharu", "Malaysia"), ("Kuala Lumpur", "Malaysia"), ("Genting", "Malaysia"),
    ("Johor", "Malaysia"), ("KTCC", "Malaysia"), ("Penang", "Malaysia"),
    ("Setia City", "Malaysia"), ("Sunway", "Malaysia"), ("Suria Sabah", "Malaysia"),
    # Thailand
    ("Bangkok", "Thailand"), ("Chiangmai", "Thailand"), ("Central World", "Thailand"),
    ("Emsphere", "Thailand"), ("Icon Siam", "Thailand"), ("Mega Bangna", "Thailand"),
    ("Pattaya", "Thailand"), ("Suwannaphum", "Thailand"), ("One Bangkok", "Thailand"),
    # South Korea
    ("Seoul", "South Korea"), ("Lotte World", "South Korea"), ("Gangnam", "South Korea"),
    ("Shinsegae", "South Korea"), ("Starfield", "South Korea"),
    # Spain
    ("Valencia", "Spain"), ("Madrid", "Spain"), ("Preciados", "Spain"),
    ("Paseo de Gracia", "Spain"), ("Puerto Banús", "Spain"), ("Bilbao", "Spain"),
    ("Sevilla", "Spain"), ("Goya", "Spain"), ("San Sebastian", "Spain"), ("Serrano", "Spain"),
    ("Gran Vía", "Spain"),
    # Japan
    ("Takashimaya", "Japan"), ("Osaka", "Japan"), ("Fukuoka", "Japan"), ("Ginza", "Japan"),
    ("Kyoto", "Japan"), ("Sapporo", "Japan"), ("Harajuku", "Japan"), ("Tokyo", "Japan"),
    ("Shibuya", "Japan"), ("Yokohama", "Japan"), ("KIX", "Japan"), ("HP Mitsui", "Japan"),
    # Singapore
    ("ION Orchard", "Singapore"), ("Marina Bay Sands", "Singapore"), ("Vivo City", "Singapore"),
    ("Waterway Point", "Singapore"),
    # Belgium / Netherlands / Scandinavia
    ("Brussels", "Belgium"), ("Antwerpen", "Belgium"),
    ("Amsterdam", "Netherlands"), ("Rotterdam", "Netherlands"), ("Utrecht", "Netherlands"),
    ("Copenhagen", "Denmark"), ("Oslo", "Norway"), ("Smålandsgatan", "Sweden"),
    # India / Mexico / Taiwan / Hong Kong / Middle East
    ("New Delhi", "India"), ("DLF", "India"), ("PMC BLR", "India"),
    ("Perisur", "Mexico"), ("Cancún", "Mexico"), ("Santa Fe", "Mexico"),
    ("Satelite", "Mexico"), ("Monterrey", "Mexico"),
    ("Taipei", "Taiwan"), ("Taichung", "Taiwan"), ("HP Gloria", "Taiwan"),
    ("Mira Place", "Hong Kong"),
    ("Dammam", "Saudi Arabia"), ("Mall of Arabia", "Saudi Arabia"),
    # A second pass on names that needed more digging
    ("Dataran Pahlawan", "Malaysia"), ("East Coast Mall", "Malaysia"), ("TRX", "Malaysia"),
    ("Westfield MON", "Netherlands"),  # Westfield Mall of the Netherlands
    ("Chersmide", "Australia"),  # likely "Chermside" (Westfield Chermside, Brisbane)
    ("Mariahilferstrasse", "Austria"),  # the Vienna shopping street
    ("Swatch Bruges", "Belgium"),
    ("Centre Comercial Illa", "Spain"),  # L'Illa Diagonal, Barcelona
    ("Panorama Mall", "Turkey"),  # Panorama is a Turkish mall chain -- least certain of these
]


def region_swatch(store_name: str) -> str:
    for keyword, region in _SWATCH_KEYWORDS:
        if keyword.lower() in store_name.lower():
            return region
    return "Other"


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
    "swatch": dict(csv="Swatch.csv", region_fn=region_swatch),
}

RAW_COLUMNS = ["col1", "store_id", "store_name", "count_in", "count_out", "ts"]
