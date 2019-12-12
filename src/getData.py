
import pandas as pd
from urllib.request import urlretrieve
import requests
import json
import shutil
from os import listdir
from os.path import isfile, join
import re


def getData(country):
    '''
    This function returns a dataframe with all the records for the specified country (it is
    also saved in the 'dataset' folder).
    Furthermore, it downloads all the audio recordings from the API to the 'dataset/recordings' folder.
    https://www.xeno-canto.org/help/search
    https://www.xeno-canto.org/explore/api
    '''
    url = f'https://www.xeno-canto.org/api/2/recordings?query=cnt:{country}'
    res = requests.get(url)
    res_dict = json.loads(res.text)
    df = pd.DataFrame()
    errors = []
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
                f"./dataset/recordings/{res_dict_page['recordings'][j]['gen']}-{res_dict_page['recordings'][j]['sp']}_{id}.mp3")
            except:
                errors.append(id)
                print(f'Download failed for id: {id}')
                continue
    # Drop recordings that failed to download from the dataframe:
    indexsToDrop = []
    for e in errors:
        for i in range(len(df)):
            if df.at[i,'id']== int(e):
                indexsToDrop.append(i)
    df.drop(indexsToDrop, axis=0, inplace=True)
    # Reordering columns:
    df = df[['id', 'en','gen', 'sp', 'ssp', 'loc', 'cnt', 'lat', 'lng', 'alt', 'date', 'time',
    'also', 'bird-seen', 'type', 'url', 'file', 'file-name','length', 'lic','playback-used',
    'q','rec', 'rmk', 'sono', 'uploaded']]
    df.to_csv(f'./dataset/birds_{country}.csv', index=False)
    return df


def selectSpecies(speciesArray, DF):
    '''
    Extract the selected species from the dataframe and save the requested audios in a separate folder.
    Returns the cleaned dataframe.
    '''
    # For the moment, I remove the rows in which it is recorded that different species sing at the same time.
    DFunique = DF[DF.also == "['']"]
    # Getting ids for the selected species:
    selectedIds = []
    for e in speciesArray:
        selectedIds.append(list(DFunique.id[DFunique.en==e]))
    selectedIds = [item for sublist in selectedIds for item in sublist]
    mypath = './dataset/recordings'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for file in files:
        if int(re.findall('[0-9]+.',file)[0][:-1]) in selectedIds:
            shutil.copy(f'./dataset/recordings/{file}', './dataset/recordings/stage-1')
    indexsSelected = [i for i in range(len(DFunique)) if DFunique['id'].iloc[i] in selectedIds]
    finalDF = DFunique.iloc[indexsSelected]
    finalDF.to_csv(f'./dataset/birds_spain_selected.csv', index=False)
    return finalDF

