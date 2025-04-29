import requests
from Logger.Logger import mylogger
from Keypass.key_pass import API_RTE

def getToken(APIname:str, logger=False):
    token_url = "https://digital.iservices.rte-france.com/token/oauth/"
    headers = {
        'Authorization': f'Basic {API_RTE[APIname]["key"]}',
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    # Send request
    response = requests.post(token_url, headers=headers)
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
    print(getToken(APIname="Generation Forecast", logger=True))
