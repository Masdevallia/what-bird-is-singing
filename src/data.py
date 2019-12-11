
from urllib.request import urlretrieve
import requests
import json
import pandas as pd


def getData(country):
    '''
    This function returns a dataframe with all records for the specified country.
    Furthermore, it downloads all the audio recordings to the folder 'dataset'.
    https://www.xeno-canto.org/help/search
    https://www.xeno-canto.org/explore/api
    '''
    url = f'https://www.xeno-canto.org/api/2/recordings?query=cnt:{country}'
    res = requests.get(url)
    res_dict = json.loads(res.text)
    df = pd.DataFrame()
    # errors = []
    for i in range(1,res_dict['numPages']+1):
        url_page = f'https://www.xeno-canto.org/api/2/recordings?query=cnt:{country}&page={i}'
        res_page = requests.get(url_page)
        res_dict_page = json.loads(res_page.text)
        df_page = pd.DataFrame(res_dict_page['recordings'])
        df = pd.concat([df,df_page])
        for j in range(len(res_dict_page['recordings'])):
            id = res_dict_page['recordings'][j]['id']
            url = f'https://www.xeno-canto.org/{id}/download'
            print(f'Page: {i}, id: {id}, url: {url}')
            try:
                urlretrieve(url,
                f"./dataset/{res_dict_page['recordings'][j]['gen']}-{res_dict_page['recordings'][j]['sp']}_{id}.mp3")
            except:
                # errors.append(id)
                print(f'Download failed for id: {id}')
                continue
    # Reordering the columns of the dataframe:
    df = df[['id', 'en','gen', 'sp', 'ssp', 'loc', 'cnt', 'lat', 'lng', 'alt', 'date', 'time',
    'also', 'bird-seen', 'type', 'url', 'file', 'file-name','length', 'lic','playback-used',
    'q','rec', 'rmk', 'sono', 'uploaded']]
    df.to_csv(f'./birds_{country}.csv', index=False)
    return df






# country = 'spain'
# gen = 'Cuculus'
# url = f'https://www.xeno-canto.org/api/2/recordings?query=cnt:{country}'
# url = f'https://www.xeno-canto.org/api/2/recordings?query=cnt:{country}+gen:{gen}'
# url = f'https://www.xeno-canto.org/api/2/recordings?query=cnt:{country}&page=5'

# Convendría que also esté vacío... Si no hay otros pájaros de fondo...
# 'also': ['']

# resjson['numRecordings']
# resjson['numRecordings']
# resjson['numSpecies']
# resjson['page']
# resjson['numPages']
# resjson['recordings']