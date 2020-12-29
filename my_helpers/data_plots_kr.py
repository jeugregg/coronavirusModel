# built-in libs
import os
import datetime
from urllib.request import Request, urlopen
from xml.etree import ElementTree
import io
import json

# third party libs
import pandas as pd
import numpy as np
import requests

# project libs
import settings
from my_helpers.dates import add_days
from my_helpers.dates import create_date_range_lim
from my_helpers.model import calc_sum_mobile
from my_helpers.model import calc_rt_from_sum
from my_helpers.model import NB_DAYS_CV
from my_helpers.utils import clean_file


# DEFINITONS 
PATH_TO_SAVE_DATA = settings.PATH_TO_SAVE_DATA

PATH_DF_FEAT_KR = PATH_TO_SAVE_DATA + '/' + 'df_feat_kr.csv'
PATH_DF_METEO_RAW_KR = os.path.join(PATH_TO_SAVE_DATA, 'df_meteo_raw_kr.csv')
PATH_DF_METEO_KR = os.path.join(PATH_TO_SAVE_DATA, 'df_meteo_kr.csv')
DATE_FIRST_CASES_GOUV_KR = '2020-02-01' # First data date in Gouv KR
DATE_FIRST_FEAT_OK_KR = '2020-04-03' # First data age/cases/meteo for features
URL_API_CASES_KR ='http://openapi.data.go.kr/openapi/service/rest/Covid19/' + \
    'getCovid19InfStateJson'
URL_API_AGE_KR = 'http://openapi.data.go.kr/openapi/service/rest/Covid19/' + \
    'getCovid19GenAgeCaseInfJson'
URL_API_AREA_KR = 'http://openapi.data.go.kr/openapi/service/rest/Covid19/' + \
    'getCovid19SidoInfStateJson'
API_KEY_KR = 'vdvTqgH%2ByZyoebTbIuQVedRNSnB9aP0IuNFfD4uIRnhALu4%2' + \
    'FUkxCDZSHp2Qx2S4IOfN3P3nJCQJbTbxk%2FdMAlA%3D%3D'

URL_GEOJSON_AREA_KR = os.path.join(PATH_TO_SAVE_DATA, 'sources', 
                                   'skorea-provinces-geo-simple.json')

GUBUN_AGE_DICT_KR = {
    "0-9": np.mean([0, 9]),
    "10-19": np.mean([10, 19]),
    "20-29": np.mean([20, 29]),
    "30-39": np.mean([30, 39]),
    "40-49": np.mean([40, 49]),
    "50-59": np.mean([50, 59]),
    "60-69": np.mean([60, 69]),
    "70-79": np.mean([70, 79]),
    "80 이상": 85
}

DICT_NBC = dict()
DICT_AGE_POS = dict()
DICT_POS_AGE = dict()
LIST_NBC = list()
for age_curr in GUBUN_AGE_DICT_KR.values():
    label_curr = f"nbC_{age_curr}"
    LIST_NBC.append(label_curr)
    DICT_NBC[label_curr] = float(age_curr)
    DICT_AGE_POS[float(age_curr)] = f"pos_{age_curr}" 
    DICT_POS_AGE[f"pos_{age_curr}"] = float(age_curr)

LIST_AREA = ["Seoul",
"Busan",
"Daegu",
"Incheon",
"Gwangju",
"Daejeon",
"Ulsan",
"Sejong",
"Gyeonggi",
"Gangwon",
"Chungbuk",
"Chungnam",
"Jeonbuk",
"Jeonnam",
"Gyeongbuk",
"Gyeongnam",
"Jeju"]


DICT_AREA = {
    'Jeju': "Jeju",
    'Gyeongsangnam-do': "Gyeongnam",
    'Gyeongsangbuk-do': "Gyeongbuk",
    'Jeollanam-do': "Jeonnam",
    'Jeollabuk-do': "Jeonbuk",
    'Chungcheongnam-do': "Chungnam",
    'Chungcheongbuk-do': "Chungbuk",
    'Gangwon-do': "Gangwon",
    'Gyeonggi-do': "Gyeonggi",
    'Sejong': "Sejong",
    'Ulsan': "Ulsan",
    'Daejeon': "Daejeon",
    'Gwangju': "Gwangju",
    'Incheon': "Incheon",
    'Daegu': "Daegu",
    'Busan': "Busan",
    'Seoul': "Seoul"
}
    
# meteo
URL_METEO_VC = "https://weather.visualcrossing.com/" + \
    "VisualCrossingWebServices/rest/services/weatherdata/history" 
API_KEY_METEO_VC = "7XNH4XB897R3PGSKJAKU7GGFL"

DICT_COL_METEO = {"maxt": "T_max",
                  "mint": "T_min",
                  "humidity": "H_mean" ,
                  'wspd': "W_speed"
                 }
LIST_COL_METEO = list(DICT_COL_METEO.values())

# maps
with open(URL_GEOJSON_AREA_KR) as f:
    GEOJSON_KR = json.load(f)
    
LIST_NAME_GEOJSON = \
    [feat_curr["properties"]["NAME_1"] for feat_curr in GEOJSON_KR['features']]

LIST_AREA_GEOJSON = [DICT_AREA[area] for area in LIST_NAME_GEOJSON]

LIST_SUM_GEOJSON = [f"sum_{DICT_AREA[area]}" for area in LIST_NAME_GEOJSON]
LIST_RT_GEOJSON = [f"Rt_{DICT_AREA[area]}" for area in LIST_NAME_GEOJSON]

LAT_LON_KR =  {'lat':  36, 'lon': 128}
ZOOM_KR = 5.5
# HELPERS

# Utils
def update_append(df1, df2):
    '''
    Update existing rows and append new rows of 2 DataFrames
    
    df2 replace rows in df1 and add new rows not in df1 to df1 
    return a new df1
    
    '''
    df1 = df1.copy()
    df2 = df2.copy()
    index1 = df1.index
    index2 = df2.index
    
    index_update = \
        [index_curr for index_curr in index2 if index_curr in index1]
    
    index_append = \
        [index_curr for index_curr in index2 if index_curr not in index1]
    
    
    if index_update != []:
        print("updating...")
        df1.update(df2.loc[index_update])
        
    if index_append != []:
        print("appending...")
        df1 = df1.append(df2.loc[index_append], verify_integrity=True)
    
    return df1

# data plot korea
def connect_api_kr(url, date_req_start, date_req_end):
    
    date_req_start = date_req_start.replace("-","")
    date_req_end = date_req_end.replace("-","")
    
    queryParams = f'?serviceKey={API_KEY_KR}' + \
                    f'&startCreateDt={date_req_start}' + \
                    f'&endCreateDt={date_req_end}'
    
    print(url + queryParams)
    
    request = Request(url + queryParams)
    request.get_method = lambda:'GET'
    response_body = urlopen(request).read()
    
    return response_body
    
    
def connect_api_cases_kr(date_req_start, date_req_end):
    '''
    Get from URL API Gouv KR cases between 2 dates
    '''
    return connect_api_kr(URL_API_CASES_KR, date_req_start, date_req_end)

def connect_api_age_kr(date_req_start, date_req_end):
    '''
    Get from URL API Gouv KR by age between 2 dates
    '''
    return connect_api_kr(URL_API_AGE_KR, date_req_start, date_req_end)

def connect_api_area_kr(date_req_start, date_req_end):
    '''
    Get from URL API Gouv KR by area between 2 dates
    '''
    return connect_api_kr(URL_API_AREA_KR, date_req_start, date_req_end)

def convert_xml_area_kr(response_body):
    '''
    Convert into DataFrame XMl reponse from URL API Gouv KR cases by Area
    '''
    root = ElementTree.XML(response_body)
    items = root.find("body").find('items').findall("item")
    print("nb. new items: ", len(items))
    if len(items) == 0:
        print("No update.")
        return None
    
    df_area_kr = pd.DataFrame(columns=["date"], 
                                  index=[])
    for item in items:
        # area
        gubunEn = item.find("gubunEn")
        if gubunEn.text not in DICT_AREA.keys():
            continue
            
        # date
        stateDt = item.find("createDt")
        #print("stateDt:", stateDt.text)
        str_date = f'{stateDt.text[0:10]}'

        # positive cases
        incDec = item.find("incDec")
        
        # add to dataFrame
        ser_curr = pd.Series({"date": str_date, 
                       DICT_AREA[gubunEn.text]:  int(incDec.text)})
        df_area_kr = df_area_kr.append(ser_curr, ignore_index=True)

    # clean one row per date 
    df_area_kr = df_area_kr.groupby("date").max()
    df_area_kr.index.name=""
    df_area_kr.index = pd.to_datetime(df_area_kr.index)
    return df_area_kr


def convert_xml_to_df_feat_kr(response_body):
    '''
    Convert into DataFrame XMl reponse from URL API Gouv KR cases
    '''
    root = ElementTree.XML(response_body)
    #ElementTree.dump(root)
    items = root.find("body").find('items').findall("item")
    print("nb. new items: ", len(items))
    if len(items) == 0:
        print("No update.")
        return None
    
    df_feat_kr_tmp = pd.DataFrame(columns=["date", "nb_cases", "nb_tests"], 
                                  index=[])
    for item in items:
        # date
        stateDt = item.find("stateDt")
        str_date = \
            f'{stateDt.text[0:4]}-{stateDt.text[4:6]}-{stateDt.text[6:8]}'

        # nb_cases (total)
        decideCnt = item.find("decideCnt")

        # nb test (total)
        accExamCompCnt = item.find("accExamCompCnt")
        if (accExamCompCnt is None):
            accExamCompCnt_val = 0
        else:
            accExamCompCnt_val = accExamCompCnt.text

        # nb death (total)
        deathCnt = item.find("deathCnt")
        if (deathCnt is None):
            deathCnt_val = 0
        else:
            deathCnt_val = deathCnt.text

        ser_curr = pd.Series(dict(date=str_date, 
                       nb_cases=int(decideCnt.text), 
                       nb_tests=int(accExamCompCnt_val),
                       nb_deaths=int(deathCnt_val)))

        df_feat_kr_tmp = df_feat_kr_tmp.append(ser_curr, ignore_index=True)

    # clean one row per date 
    df_feat_kr_tmp = df_feat_kr_tmp.groupby("date").max()
    df_feat_kr_tmp.index.name=""
    df_feat_kr_tmp["date"] = df_feat_kr_tmp.index
    df_feat_kr_tmp.index = pd.to_datetime(df_feat_kr_tmp.index)
    return df_feat_kr_tmp

def convert_xml_age_kr(response_body):
    root = ElementTree.XML(response_body)
    #ElementTree.dump(root)
    items = root.find("body").find('items').findall("item")
    print("nb. new items: ", len(items))
    if len(items) == 0:
        print("No update.")
        return None
    
    df_age_kr = pd.DataFrame(columns=["date"], 
                                  index=[])
    for item in items:
        
        # age cat 
        gubun = item.find("gubun")
        if gubun.text not in GUBUN_AGE_DICT_KR.keys():
            continue
        
        age_cat = GUBUN_AGE_DICT_KR[gubun.text]
        
        # date
        stateDt = item.find("createDt")
        #print("stateDt:", stateDt.text)
        str_date = f'{stateDt.text[0:10]}'

        # nb_cases (total)
        confCase = item.find("confCase")

        # add to dataFrame
        ser_curr = pd.Series({"date": str_date, 
                       f"nbC_{age_cat}":  int(confCase.text)})
        df_age_kr = df_age_kr.append(ser_curr, ignore_index=True)
        
    # clean dates and interpolate if NaN :
    df_age_kr = df_age_kr.groupby("date")[LIST_NBC].sum()
    df_age_kr.index = pd.to_datetime(df_age_kr.index)
    df_age_kr = df_age_kr.resample('1D').asfreq()
    df_age_kr = df_age_kr.interpolate(method='linear', 
                                      limit_direction='forward', 
                                      axis=0)
    df_age_kr["nbC_age"] = df_age_kr.sum(axis=1)
        
        
    return df_age_kr
    

# check update ?
def check_update_df_feat_kr(date_now=None, force_update=False):
    
    if date_now is None:
        date_now = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if force_update:
        return True, DATE_FIRST_CASES_GOUV_KR, \
                DATE_FIRST_CASES_GOUV_KR, date_now
    
    flag_update = True # update to be done ?
    flag_update_age = True
    
    if os.path.isfile(PATH_DF_FEAT_KR):
        df_feat_kr = pd.read_csv(PATH_DF_FEAT_KR)
        date_req_start = add_days(df_feat_kr["date"].max(), 1)
        date_req_start_age = \
            df_feat_kr[df_feat_kr["age_pos"].isna()]["date"].max()
    else:
        date_req_start = DATE_FIRST_CASES_GOUV_KR
        date_req_start_age = DATE_FIRST_CASES_GOUV_KR
    
    if date_req_start >= date_now:
            flag_update = False
            
    if date_req_start_age >= date_now:
            flag_update = False
    
    if not flag_update:
        date_req_start = None
        
    if not flag_update_age:
        date_req_start_age = None
    
    if (flag_update | flag_update_age):
        date_req_end = date_now
    else:
        date_req_end = None
    print("Updating Data KR...")
    print("update cases : ", flag_update)
    print("update age : ", flag_update_age)
    print("date_req_start: ", date_req_start)
    print("date_req_start_age: ", date_req_start_age)
    print("date_req_end: ", date_req_end)
    return flag_update, flag_update_age, \
        date_req_start, date_req_start_age, date_req_end
        

def get_update_df_feat_kr(date_now=None, force_update=False):
    '''
    Get only new data cases Gouv KR
    '''
    flag_update, flag_update_age, date_req_start, date_req_start_age, \
        date_req_end = check_update_df_feat_kr(date_now, force_update)
    
    
    if flag_update:
        response_body = connect_api_cases_kr(date_req_start, date_req_end)
        df_feat_kr_tmp = convert_xml_to_df_feat_kr(response_body)
        
        # add day_num
        df_feat_kr_tmp['day_num'] = \
        df_feat_kr_tmp["date"].astype(np.datetime64).dt.strftime("%w")
        
        # add areas
        response_body = connect_api_area_kr(date_req_start, date_req_end)
        df_area_kr = convert_xml_area_kr(response_body)
        df_feat_kr_tmp = df_feat_kr_tmp.join(df_area_kr)
        
        # add meteo
        date_req_start_meteo = max(date_req_start, DATE_FIRST_FEAT_OK_KR)
        date_req_start_meteo = max(date_req_start_meteo, date_req_start_age)
        df_meteo = connect_api_meteo(date_req_start_meteo, 
                                     date_req_end)
        # save meteo
        df_meteo.to_csv(PATH_DF_METEO_KR, index=False)
        df_feat_kr_tmp = df_feat_kr_tmp.join(df_meteo)
        
    else:
        df_feat_kr_tmp = None
    
    if flag_update_age:
        # age
        response_body = connect_api_age_kr(date_req_start_age, date_req_end)

        df_age_kr = convert_xml_age_kr(response_body)
        if df_age_kr is not None:
            if (df_feat_kr_tmp is None):
                # if update for age but not for cases, have to load old df_feat 
                df_feat_kr = load_df_feat_kr()
                df_feat_kr_tmp = df_feat_kr.loc[df_age_kr.index]

            if LIST_NBC[0] not in df_feat_kr_tmp.columns:
                print("joining...")
                df_feat_kr_tmp = df_feat_kr_tmp.join(df_age_kr)
            else:
                print("updating...")
                df_feat_kr_tmp.update(df_age_kr)
        
    return df_feat_kr_tmp


def update_df_feat_kr(date_now=None, force_update=False, force_calc=False):
    '''
    Update Df Feat with new cases from Gouv KR
    force_update : to replace existing file
    force_calc : to force redo final calculation 
    '''
    # get just new data 
    df_feat_kr_tmp = get_update_df_feat_kr(date_now, force_update)
    
    # what to do with new data ? : force to be updated totally ?
    if force_update:
        df_feat_kr = df_feat_kr_tmp
    else:
        if os.path.isfile(PATH_DF_FEAT_KR):
            df_feat_kr = load_df_feat_kr()
            if df_feat_kr_tmp is not None:
                df_feat_kr = update_append(df_feat_kr, df_feat_kr_tmp)
        else:
            df_feat_kr = df_feat_kr_tmp
        
        if (df_feat_kr_tmp is None) & (not force_calc):
            return df_feat_kr
        
        # calculate derivative values    
        df_feat_kr["pos"] = df_feat_kr["nb_cases"].diff()
        df_feat_kr["test"] = df_feat_kr["nb_tests"].diff()
    

        # calculate sum-cases
        ser_sum = calc_sum_mobile(df_feat_kr["date"], df_feat_kr["pos"], 
                                  NB_DAYS_CV)
        ser_sum.name = "sum_cases"
        df_feat_kr.drop(columns=["sum_cases"], inplace=True, errors="ignore")
        df_feat_kr = df_feat_kr.join(ser_sum)

        # calculate sum-tests
        ser_sum_t = calc_sum_mobile(df_feat_kr["date"], df_feat_kr["test"], 
                                  NB_DAYS_CV)
        ser_sum_t.name = "sum_tests"
        df_feat_kr.drop(columns=["sum_tests"], inplace=True, errors="ignore")
        df_feat_kr = df_feat_kr.join(ser_sum_t)
        
        # calculate Rt country : Rt
        ser_rt = calc_rt_from_sum(df_feat_kr["sum_cases"], NB_DAYS_CV)
        ser_rt.name = "Rt"
        df_feat_kr.drop(columns=["Rt"], inplace=True, errors="ignore")
        df_feat_kr = df_feat_kr.join(ser_rt)
        
        # caculation sums by area
        for area_curr in LIST_AREA:
            # calculate sum-cases by area : col= sum_"area"
            ser_sum = calc_sum_mobile(df_feat_kr["date"], 
                                      df_feat_kr[area_curr], 
                                          NB_DAYS_CV)
            ser_sum.name = f"sum_{area_curr}"
            df_feat_kr.drop(columns=[ser_sum.name], inplace=True, 
                            errors="ignore")
            df_feat_kr = df_feat_kr.join(ser_sum)

            # calculate Rt by area : col= Rt_"area"
            ser_rt = calc_rt_from_sum(df_feat_kr[f"sum_{area_curr}"], 
                                      NB_DAYS_CV)
            ser_rt.name = f"Rt_{area_curr}"
            df_feat_kr.drop(columns=[ser_rt.name], inplace=True, 
                            errors="ignore")
            df_feat_kr = df_feat_kr.join(ser_rt)

        # positive rate over 14 days calculation
        df_feat_kr["rate_pos"] = \
            100*df_feat_kr["sum_cases"] / df_feat_kr["sum_tests"]
        
        # age  calculation
        for nbC_curr in LIST_NBC:
            df_feat_kr[DICT_AGE_POS[DICT_NBC[nbC_curr]]] = \
                df_feat_kr[nbC_curr].diff()
        
        df_feat_kr["age_pos"] = 0
        df_feat_kr["daily_age"] = df_feat_kr["nbC_age"].diff()
        
        for age_curr in DICT_AGE_POS.keys():
            df_feat_kr["age_pos"] += df_feat_kr[DICT_AGE_POS[age_curr]]*age_curr
        
        df_feat_kr["age_pos"] /= df_feat_kr["daily_age"]
    
    # save
    if df_feat_kr is not None:
        clean_file(PATH_DF_FEAT_KR, flag_copy=True)
        df_feat_kr.to_csv(PATH_DF_FEAT_KR, index=False)

    return df_feat_kr

def load_df_feat_kr():
    '''
    Load DataFrame for features of South Korea
    '''
    df_feat_kr = pd.read_csv(PATH_DF_FEAT_KR)
    df_feat_kr.index = df_feat_kr["date"]
    df_feat_kr.index = pd.to_datetime(df_feat_kr.index)
    
    return df_feat_kr

def connect_api_meteo(date_req_start, date_req_end):
    
    def fun_date_meteo(str_in):
        return f"{str_in[6:10]}-{str_in[0:2]}-{str_in[3:5]}"
    def create_query(date_req_start, date_req_end):
        return "?aggregateHours=24" + \
        "&combinationMethod=aggregate" + \
        f"&startDateTime={date_req_start}T00%3A00%3A00" + \
        f"&endDateTime={date_req_end}T00%3A00%3A00" + \
        "&maxStations=-1" + \
        "&maxDistance=-1" + \
        "&shortColumnNames=true" + \
        "&sendAsDatasource=true" + \
        "&contentType=csv" + \
        "&unitGroup=metric" + \
        "&locationMode=array" + \
        f"&key={API_KEY_METEO_VC}" + \
        "&dataElements=all" + \
        "&locations=Seoul%20south%20korea" + \
        "%7Cbusan%20south%20korea" + \
        "%7CDaegu%20South%20Korea"
    
    list_dates_start, list_dates_end = create_date_range_lim(date_req_start, 
                                                         date_req_end) 
    df_meteo_kr = None
    for date_start, date_end in zip(list_dates_start, list_dates_end):  
        
        queryParams = create_query(date_start, date_end)
        print(URL_METEO_VC + queryParams)

        # Requests
        req = requests.get(URL_METEO_VC + queryParams).content
        df_meteo_kr_tmp = pd.read_csv(io.StringIO(req.decode('utf-8')), sep=",", 
            low_memory=False)
        if df_meteo_kr is None:
            df_meteo_kr = df_meteo_kr_tmp
        else:
            df_meteo_kr = df_meteo_kr.append(df_meteo_kr_tmp)
    
    df_meteo_kr.to_csv(PATH_DF_METEO_RAW_KR, index=False)
    
    # reformat cols
    df_meteo_kr.rename(columns=DICT_COL_METEO, inplace=True)
    df_meteo_kr["date"] = df_meteo_kr["datetime"].apply(fun_date_meteo)
    # calculate mean for output
    df_meteo_kr = df_meteo_kr.groupby("date")[LIST_COL_METEO].mean()
    df_meteo_kr.index = pd.to_datetime(df_meteo_kr.index)
    
    return df_meteo_kr




