"""Region-aware holiday calendars, 2022-2026.

Movable lunar/ecclesiastical dates (Bayram, Chinese New Year, Orthodox/Western
Easter, Mid-Autumn, Dragon Boat) are hardcoded per-year approximations (fine
for feature engineering, not for liturgical accuracy). Rule-based movable
holidays (Mother's Day, Black Friday, Thanksgiving, Clean Monday) are computed
from date arithmetic.
"""
from datetime import date, timedelta

YEARS = [2022, 2023, 2024, 2025, 2026]


def _nth_weekday(year, month, weekday, n):
    """n-th (1-indexed) given weekday (Mon=0) of a month."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    d += timedelta(days=offset + 7 * (n - 1))
    return d


def _last_weekday(year, month, weekday):
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


# ---- hardcoded movable dates -------------------------------------------------
RAMAZAN_DAY1 = {2022: date(2022, 5, 2), 2023: date(2023, 4, 21), 2024: date(2024, 4, 10),
                2025: date(2025, 3, 30), 2026: date(2026, 3, 20)}
KURBAN_DAY1 = {2022: date(2022, 7, 9), 2023: date(2023, 6, 28), 2024: date(2024, 6, 16),
               2025: date(2025, 6, 6), 2026: date(2026, 5, 27)}
CNY_DAY1 = {2022: date(2022, 2, 1), 2023: date(2023, 1, 22), 2024: date(2024, 2, 10),
            2025: date(2025, 1, 29), 2026: date(2026, 2, 17)}
MID_AUTUMN = {2022: date(2022, 9, 10), 2023: date(2023, 9, 29), 2024: date(2024, 9, 17),
              2025: date(2025, 10, 6), 2026: date(2026, 9, 25)}
DRAGON_BOAT = {2022: date(2022, 6, 3), 2023: date(2023, 6, 22), 2024: date(2024, 6, 10),
               2025: date(2025, 5, 31), 2026: date(2026, 6, 19)}
ORTHODOX_EASTER = {2022: date(2022, 4, 24), 2023: date(2023, 4, 16), 2024: date(2024, 5, 5),
                   2025: date(2025, 4, 20), 2026: date(2026, 4, 12)}
WESTERN_EASTER = {2022: date(2022, 4, 17), 2023: date(2023, 4, 9), 2024: date(2024, 3, 31),
                  2025: date(2025, 4, 20), 2026: date(2026, 4, 5)}


def build_region_events(region: str):
    """Return list of (date, label, kind) for a region across YEARS.
    kind in {'eve','day','event'} -- used to build proximity + flag features."""
    ev = []

    def add_window(day1, n_days, label):
        ev.append((day1 - timedelta(days=1), f"{label}_eve", "eve"))
        for i in range(n_days):
            ev.append((day1 + timedelta(days=i), f"{label}_day{i+1}", "day"))

    for y in YEARS:
        ev.append((date(y, 1, 1), "new_year_day", "day"))
        ev.append((date(y, 12, 31), "new_year_eve", "eve"))
        ev.append((date(y, 2, 14), "valentines", "event"))
        ev.append((_nth_weekday(y, 5, 6, 2), "mothers_day", "event"))  # 2nd Sunday of May
        ev.append((_last_weekday(y, 11, 4), "black_friday", "event"))  # last Friday of Nov

        if region in ("Turkey", "Russia", "Georgia", "Albania", "Bosnia", "Egypt",
                      "Hungary", "Macedonia"):
            add_window(RAMAZAN_DAY1[y], 3, "ramazan_bayrami")
            add_window(KURBAN_DAY1[y], 4, "kurban_bayrami")
            ev.append((date(y, 3, 8), "womens_day", "event"))
            ev.append((date(y, 4, 23), "national_sovereignty_day", "day"))
            ev.append((date(y, 5, 1), "labour_day", "day"))
            ev.append((date(y, 5, 19), "youth_sports_day", "day"))
            ev.append((date(y, 8, 30), "victory_day", "day"))
            ev.append((date(y, 10, 29), "republic_day", "day"))

        if region == "China":
            add_window(CNY_DAY1[y], 7, "cny")
            ev.append((date(y, 5, 1), "labour_day_golden_week_start", "day"))
            for i in range(5):
                ev.append((date(y, 5, 1) + timedelta(days=i), "labour_golden_week", "day"))
            for i in range(7):
                ev.append((date(y, 10, 1) + timedelta(days=i), "national_day_golden_week", "day"))
            ev.append((MID_AUTUMN[y], "mid_autumn", "event"))
            ev.append((DRAGON_BOAT[y], "dragon_boat", "event"))
            ev.append((date(y, 5, 20), "520", "event"))
            ev.append((date(y, 6, 18), "618", "event"))
            ev.append((date(y, 11, 11), "double11", "event"))
            ev.append((date(y, 12, 12), "double12", "event"))

        if region in ("USA", "Canada"):
            thanksgiving = _nth_weekday(y, 11, 3, 4)  # 4th Thursday of Nov
            ev.append((thanksgiving, "thanksgiving", "day"))
            ev.append((thanksgiving + timedelta(days=1), "black_friday_us", "event"))
            ev.append((date(y, 12, 25), "christmas", "day"))
            ev.append((date(y, 7, 4), "july_4th", "day") if region == "USA" else (date(y, 7, 1), "canada_day", "day"))

        if region in ("Hong Kong", "Macau", "Taiwan"):
            add_window(CNY_DAY1[y], 3, "cny")
            ev.append((MID_AUTUMN[y], "mid_autumn", "event"))
            for i in range(5):
                ev.append((date(y, 5, 1) + timedelta(days=i), "labour_golden_week", "day"))
            ev.append((date(y, 12, 25), "christmas", "day"))

        if region == "Greece":
            easter = ORTHODOX_EASTER[y]
            for i, lbl in [(-2, "holy_friday"), (-1, "holy_saturday"), (0, "easter_sunday"), (1, "easter_monday")]:
                ev.append((easter + timedelta(days=i), lbl, "day"))
            ev.append((easter - timedelta(days=48), "clean_monday", "day"))
            ev.append((date(y, 8, 15), "assumption_of_mary", "day"))
            ev.append((date(y, 3, 25), "greek_independence_day", "day"))
            ev.append((date(y, 12, 25), "christmas", "day"))
            for i in range(11):
                ev.append((date(y, 12, 20) + timedelta(days=i), "christmas_lottery_period", "event"))

        if region == "Peru":
            ev.append((date(y, 7, 28), "fiestas_patrias", "day"))
            ev.append((date(y, 7, 29), "fiestas_patrias", "day"))
            ev.append((date(y, 12, 24), "noche_buena", "eve"))
            ev.append((date(y, 12, 25), "christmas", "day"))
            for i in range(43):  # Feb1-Mar15 back-to-school ramp
                ev.append((date(y, 2, 1) + timedelta(days=i), "back_to_school", "event"))

        if region == "Norway":
            easter = WESTERN_EASTER[y]
            for i in range(-3, 2):
                ev.append((easter + timedelta(days=i), "easter_week", "day"))
            ev.append((date(y, 5, 1), "labour_day", "day"))
            ev.append((date(y, 5, 17), "constitution_day", "day"))
            ev.append((date(y, 12, 24), "christmas_eve", "day"))
            ev.append((date(y, 12, 25), "christmas", "day"))
            for i in range(9, 21):
                ev.append((date(y, 9, 1) + timedelta(days=i), "kulturnatt_window", "event"))

    return ev


_CACHE = {}


def get_region_events(region: str):
    if region not in _CACHE:
        _CACHE[region] = build_region_events(region)
    return _CACHE[region]


ALL_KNOWN_REGIONS = ["Turkey", "Russia", "Georgia", "Albania", "Bosnia", "Egypt", "Hungary",
                     "Macedonia", "China", "USA", "Canada", "Hong Kong", "Macau", "Taiwan",
                     "Greece", "Peru", "Norway"]
ALL_HOLIDAY_LABELS = sorted({lbl for r in ALL_KNOWN_REGIONS for _, lbl, _ in get_region_events(r)} | {"none"})
