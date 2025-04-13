from entsoe import EntsoeRawClient
import pandas as pd

client = EntsoeRawClient(api_key='<YOUR API KEY>')

start = pd.Timestamp('20171201', tz='Europe/Brussels')
end = pd.Timestamp('20180101', tz='Europe/Brussels')
country_code = 'BE'  # Belgium
country_code_from = 'FR'  # France
country_code_to = 'DE_LU' # Germany-Luxembourg
type_marketagreement_type = 'A01'
contract_marketagreement_type = 'A01'
process_type = 'A51'