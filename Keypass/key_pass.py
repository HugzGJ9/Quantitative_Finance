API_KEY_ENTSOE = 'e37c0206-5d3c-4258-84d6-323ac6044d69'
KEYPASS = {'hugo.lambert.perso@gmail.com' : 'dijyasvooloqsdbi'}#API GMAIL
API_RTE ={'Wholesale Market' : {'token_url': 'https://digital.iservices.rte-france.com/open_api/wholesale_market/v2/france_power_exchanges', 'key': 'YzZkZDc2YTItMDUyZi00Y2FhLTg0NjMtMDE5YmI3ODBmMGMxOjAyMDMzM2Q2LTgwOWUtNDQ0NS05NGE4LWM1YzQ0Mzk3ZmJiZA=='},
          'Actual Generation': {
              'token_url': 'https://digital.iservices.rte-france.com/open_api/actual_generation/v1/actual_generations_per_production_type?',
              'key': 'ODI5NzMxZjktNmI0MS00NDRlLWEyZWUtODkwMmJkNTU3ODNkOjZhMzNmODIzLTYyYTktNDMwMy05NzllLTQ5MTM0MWUwODM3ZA=='},
          'Actual Generation 15min': {
              'token_url': f'https://digital.iservices.rte-france.com/open_api/actual_generation/v1/generation_mix_15min_time_scale?',
              'key': 'ODI5NzMxZjktNmI0MS00NDRlLWEyZWUtODkwMmJkNTU3ODNkOjZhMzNmODIzLTYyYTktNDMwMy05NzllLTQ5MTM0MWUwODM3ZA=='},
          'Generation Forecast': {
              'token_url': 'https://digital.iservices.rte-france.com/open_api/generation_forecast/v2/forecasts?',
              'key': 'MWM4YzcwODgtNmViNS00NGQ1LWFlYjktOGQ1MDgyYmViYzE1OmIwZWY2MjdiLWE0NWMtNDhlYi04NzQ4LTI5YmRmODBkOGQ0ZQ=='}}

API_SUPABASE = {'DAPowerPriceFR': {'token_url': 'https://wjcimcwgdbrytgjqmfdo.supabase.co', 'key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndqY2ltY3dnZGJyeXRnanFtZmRvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDk2NTQ5MiwiZXhwIjoyMDYwNTQxNDkyfQ.9G2CNhzirAx9vSNcUZX_0qd-8TD1ruvX3zgjq4E7cN8'},
                'PPA': {'token_url': 'https://wjcimcwgdbrytgjqmfdo.supabase.co', 'key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndqY2ltY3dnZGJyeXRnanFtZmRvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDk2NTQ5MiwiZXhwIjoyMDYwNTQxNDkyfQ.9G2CNhzirAx9vSNcUZX_0qd-8TD1ruvX3zgjq4E7cN8'},
                'CALPowerPriceFR': {'token_url': 'https://wjcimcwgdbrytgjqmfdo.supabase.co', 'key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndqY2ltY3dnZGJyeXRnanFtZmRvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDk2NTQ5MiwiZXhwIjoyMDYwNTQxNDkyfQ.9G2CNhzirAx9vSNcUZX_0qd-8TD1ruvX3zgjq4E7cN8'},
                'GenerationFR': {'token_url': 'https://wjcimcwgdbrytgjqmfdo.supabase.co', 'key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndqY2ltY3dnZGJyeXRnanFtZmRvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDk2NTQ5MiwiZXhwIjoyMDYwNTQxNDkyfQ.9G2CNhzirAx9vSNcUZX_0qd-8TD1ruvX3zgjq4E7cN8'},
                'InstalledCapacityFR': {'token_url': 'https://wjcimcwgdbrytgjqmfdo.supabase.co', 'key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndqY2ltY3dnZGJyeXRnanFtZmRvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDk2NTQ5MiwiZXhwIjoyMDYwNTQxNDkyfQ.9G2CNhzirAx9vSNcUZX_0qd-8TD1ruvX3zgjq4E7cN8'},
                'ForecastGenerationFR': {'token_url': 'https://wjcimcwgdbrytgjqmfdo.supabase.co',
                                        'key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndqY2ltY3dnZGJyeXRnanFtZmRvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDk2NTQ5MiwiZXhwIjoyMDYwNTQxNDkyfQ.9G2CNhzirAx9vSNcUZX_0qd-8TD1ruvX3zgjq4E7cN8'},
                'ShapeMonthSR_FR': {'token_url': 'https://wjcimcwgdbrytgjqmfdo.supabase.co',
                                        'key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndqY2ltY3dnZGJyeXRnanFtZmRvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDk2NTQ5MiwiZXhwIjoyMDYwNTQxNDkyfQ.9G2CNhzirAx9vSNcUZX_0qd-8TD1ruvX3zgjq4E7cN8'},
                'ShapeMonthWIND_FR': {'token_url': 'https://wjcimcwgdbrytgjqmfdo.supabase.co',
                                        'key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndqY2ltY3dnZGJyeXRnanFtZmRvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDk2NTQ5MiwiZXhwIjoyMDYwNTQxNDkyfQ.9G2CNhzirAx9vSNcUZX_0qd-8TD1ruvX3zgjq4E7cN8'},
                'WeatherFR': {'token_url': 'https://wjcimcwgdbrytgjqmfdo.supabase.co', 'key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndqY2ltY3dnZGJyeXRnanFtZmRvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDk2NTQ5MiwiZXhwIjoyMDYwNTQxNDkyfQ.9G2CNhzirAx9vSNcUZX_0qd-8TD1ruvX3zgjq4E7cN8'}}
SUPABASE_pass = 'LjY+5CK3S5z_L?w'
