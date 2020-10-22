import zcrmsdk
import pandas as pd
from zcrmsdk import ZCRMException

config = {
"client_id": "1000.5LP4BFDBQ9OOODQAUC7EX3L37YBQ0H",
"client_secret": "3c65a539fa7dc4fca3e6fa1b9f1dd5de77d6b7c285",
"redirect_uri": "https://www.kncomercial.com",
"token_persistence_path": "/Users/gerardorodriguez/Documents/files/CRM/tokens/",
"currentUserEmail": "research+crm@knotion.com",
"accounts_url": "https://accounts.zoho.com"
}


def Grant_access(config):
    zcrmsdk.ZCRMRestClient.initialize(config)
    oauth_client = zcrmsdk.ZohoOAuth.get_client_instance()
    grant_token = "1000.112ae3ce436c2dd3d0847992426dca06.55d0727e515f427411df4de8f93ef6fe"
    oauth_tokens = oauth_client.generate_access_token(grant_token)
    return 0


from zcrmsdk import ZCRMRecord
def update_record(file):
    for i in range(len(file)):
        record_id = file.iloc[i, :]['Colegio ID'].split('_')[1]
        amai = file.iloc[i]['Clase AMAI']
        nivel = file.iloc[i]['Nivel Socioeconómico']
        porc = round(file.iloc[i]['Porcentaje de compatibilidad'], 2)
        try:
            record = ZCRMRecord.get_instance('Accounts', int(record_id))
            #record.set_field_value('Clase_AMAI', amai)
            #record.set_field_value('Nivel_Socio_econ_mico', nivel)
            record.set_field_value('Compatibilidad', str(porc))
            resp = record.update()
            print(resp.status_code)
            print(resp.code)
            print(resp.details)
            print(resp.message)
            print(resp.status)
        except ZCRMException as ex:
            print(ex.status_code)
            print(ex.error_message)
            print(ex.error_code)
            print(ex.error_details)
            print(ex.error_content)


def read_csv():
    Grant_access(config)
    #zcrmsdk.ZCRMRestClient.initialize(config)
    #oauth_client = zcrmsdk.ZohoOAuth.get_client_instance().get_access_token('research+crm@knotion.com')
    #print(oauth_client)

    file = pd.read_csv('Colegios_to_crm.csv')
    update_record(file)

if __name__=='__main__':
    read_csv()


