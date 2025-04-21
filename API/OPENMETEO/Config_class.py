from datetime import datetime, timedelta
class Config:
    latitude: float = 46.0
    longitude: float = 2.0
    history_days: int = 3600
    random_seed: int = 42
    start = (datetime.utcnow() - timedelta(days=history_days)).strftime("%Y-%m-%d")
    end = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
cfg = Config()

url = {'history': (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={cfg.latitude}&longitude={cfg.longitude}&start_date={cfg.start}&end_date={cfg.end}"
        "&hourly=shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,global_tilted_irradiance,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,,wind_speed_100m,wind_direction_100m,wind_gusts_10m,surface_pressure&timezone=UTC"
    ),
    'forecast': (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={cfg.latitude}&longitude={cfg.longitude}&hourly=shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,global_tilted_irradiance,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,,wind_speed_100m,wind_direction_100m,wind_gusts_10m,surface_pressure&timezone=UTC"
    )}