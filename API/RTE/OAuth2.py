import requests
from Logger.Logger import mylogger

API ={'Wholesale Market' : {'token_url': 'https://digital.iservices.rte-france.com/open_api/wholesale_market/v2/france_power_exchanges', 'key': 'YzZkZDc2YTItMDUyZi00Y2FhLTg0NjMtMDE5YmI3ODBmMGMxOjAyMDMzM2Q2LTgwOWUtNDQ0NS05NGE4LWM1YzQ0Mzk3ZmJiZA=='},
      'Actual Generation' : {'token_url': 'https://digital.iservices.rte-france.com/open_api/actual_generation/v1/actual_generations_per_production_type?', 'key': 'ODI5NzMxZjktNmI0MS00NDRlLWEyZWUtODkwMmJkNTU3ODNkOjZhMzNmODIzLTYyYTktNDMwMy05NzllLTQ5MTM0MWUwODM3ZA=='}}
def getToken(APIname:str, logger=False):
    token_url = "https://digital.iservices.rte-france.com/token/oauth/"
    data = {
        'Authorization' : f'Basic {API[APIname]["key"]}',
        'Content-Type' : 'application/x-www-form-urlencoded',
        'start_date': '2025-01-01T00:00:00+02:00',
        'end_date': '2025-04-12T22:00:00+02:00',
    }

    response = requests.post(token_url, headers=data)
    status_code = response.status_code
    if logger:
        mylogger.logger.info(f'status code {status_code}')
    info_rte_token = response.json()
    if logger:
        mylogger.logger.info(f'Info RTE token {info_rte_token}')
    token = info_rte_token['access_token']
    return token

if __name__ == '__main__':
    print(getToken(APIname="Wholesale Market", logger=True))
    print(getToken(APIname="Actual Generation", logger=True))