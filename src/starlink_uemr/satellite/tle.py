import skyfield
from skyfield.api import load, wgs84


def build_timescale():
    return load.timescale()


def load_satellite_from_tle(tle_path: str, sat_name_keyword: str):
    satellites = load.tle_file(tle_path)
    sat_name_keyword = sat_name_keyword.strip()

    # Prefer exact-name match to keep notebook scans and final selection consistent.
    exact = [sat for sat in satellites if sat.name == sat_name_keyword]
    if exact:
        return exact[0]

    sat_name_keyword_upper = sat_name_keyword.upper()

    matched = [sat for sat in satellites if sat_name_keyword_upper in sat.name.upper()]
    if not matched:
        raise ValueError(f"No satellite matched keyword: {sat_name_keyword}")

    return matched[0]


def make_observer(lat_deg: float, lon_deg: float, elev_m: float):
    return wgs84.latlon(
        latitude_degrees=lat_deg,
        longitude_degrees=lon_deg,
        elevation_m=elev_m,
    )
