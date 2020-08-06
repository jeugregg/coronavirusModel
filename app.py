# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import tensorflow as tf

import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import datetime
import math
import re
import os

# read json from http
import json
import urllib.request

# read csv from http
import io
import requests

# DEFINITIONS
PATH_TO_SAVE_DATA = "."
PATH_DF_POS_FR = PATH_TO_SAVE_DATA + '/' + 'df_pos_fr.csv'
PATH_DF_TEST_FR = PATH_TO_SAVE_DATA + '/' + 'df_test_fr.csv'
PATH_JSON_METEO_FR = PATH_TO_SAVE_DATA + '/' + 'data_meteo_fr.json'
PATH_DF_FEAT_FR = PATH_TO_SAVE_DATA + '/' + 'df_feat_fr.csv' 
PATH_GEO_DEP_FR = PATH_TO_SAVE_DATA + '/sources/geofrance/' + 'departments.csv'
PATH_MDL_SINGLE_STEP = PATH_TO_SAVE_DATA + '/' + "mdl_single_step_pos_fr"
PATH_MDL_MULTI_STEP = PATH_TO_SAVE_DATA + '/' + "mdl_multi_step_pos_fr"
URL_CSV_GOUV_FR = \
    'https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675'

NB_POS_DATE_MIN_DF_FEAT = 140227 # on 12/05/2020
date_format = "%Y-%m-%d"
# model parameters
train_split = 50
past_history = 10 # days used to predict next values in future
future_target = 3 # predict 3 days later
STEP = 1

# HELPER FUNCTIONS
# FOR DATES
def add_days(str_date_0, nb_days_CV):
    '''
    add days to string dates
    '''
    date_format = "%Y-%m-%d"
    date_last = datetime.datetime.strptime(str_date_0, date_format)
    date_start = date_last + datetime.timedelta(days=nb_days_CV)
    str_date_start = date_start.strftime(date_format)
    return str_date_start

def generate_list_dates(str_date_0, str_date_1, date_format=None):
    if date_format is None:
        date_format = "%Y-%m-%d"
    date_0 = datetime.datetime.strptime(str_date_0, date_format)
    date_1 = datetime.datetime.strptime(str_date_1, date_format)
    delta = date_1 - date_0
    if delta.days > 0:
        return [(date_0 + \
                 datetime.timedelta(days=I)).strftime(date_format) \
               for I in range(1, delta.days+1)]
    else:
        return None

def days_between(str_date_0, str_date_1):
    '''
    calculate days between 2 string dates
    '''
    date_format = "%Y-%m-%d"
    
    date_0 = datetime.datetime.strptime(str_date_0, date_format)
    date_1 = datetime.datetime.strptime(str_date_1, date_format)
    delta = date_1 - date_0
    '''date_start = date_last + datetime.timedelta(days=nb_days_CV)
    str_date_start = date_start.strftime(date_format)
    return str_date_start'''
    return delta

def get_file_date(path_to_file):
    '''
    get file modification date 
    '''
    return datetime.datetime.utcfromtimestamp(os.path.getmtime(path_to_file))

def conv_dt_2_str(dt_in):
    '''
    Convert datatime to string date
    '''
    return  dt_in.strftime("%Y-%m-%d %H:%M:%S")

# For METEO
def create_url_meteo_date(str_date):
    # str_date = 2020-05-10
    num_records_max = 10000
    return 'https://public.opendatasoft.com/api/records/1.0/search/' + \
        f'?dataset=donnees-synop-essentielles-omm&q=&rows={num_records_max}' + \
        f'&sort=date&refine.date={str_date}'

def get_data_meteo_by_date(str_date):
    '''
    get data meteo for 1 day on date str_date 
    
    example : get_data_meteo_by_date("2020-01-24")
    '''
    # download raw json object
    url = create_url_meteo_date(str_date)
    data = urllib.request.urlopen(url).read().decode()
    # parse json object
    return json.loads(data)

def get_data_meteo_by_list(list_date):
    '''
    Retrieve data meteo for a list of dates
    '''
    for I, date_curr in enumerate(list_date):
        
        data_curr = get_data_meteo_by_date(date_curr)
        
        if I:
            data_out["records"] = data_out["records"] + data_curr["records"]
        else:
            data_out = data_curr.copy()
            
    return data_out
    

def create_url_meteo(num_records, num_start=0):
    # https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm  
    return 'https://public.opendatasoft.com/api/records/1.0/search/' + \
        f'?dataset=donnees-synop-essentielles-omm&q=&rows={num_records}' + \
        f'&sort=date&start={num_start}'

def get_data_meteo(num_records, num_start=0):
    # num_records max = 10000
    # download raw json object
    url = create_url_meteo(num_records, num_start)
    data = urllib.request.urlopen(url).read().decode()
    # parse json object
    return json.loads(data)

# function about data meteo from json 
def get_data_meteo_date_list(data): 
    return np.unique([data["records"][I]["fields"]["date"][0:10] \
                for I in range(len(data["records"]))]).tolist()

def get_data_meteo_date_min(data):
    list_date = get_data_meteo_date_list(data)
    return min(list_date)[0:10]

def get_data_meteo_date_max(data):
    list_date = get_data_meteo_date_list(data)
    return max(list_date)[0:10]

def get_rec_by_date(data_meteo, date_str):
    '''
    get one date data 
    '''
    
    # date_str = '2020-05-14'
    list_rec = []
    for rec_curr in data_meteo["records"]:
        if rec_curr['fields']["date"][0:10] == date_str:
            list_rec.append(rec_curr)
    data_out = data_meteo.copy()
    data_out["records"] = list_rec
    return data_out

def select_rec_by_station(data_meteo):
    '''
    Select list of list of data by station
    '''
    list_station = [data_meteo["records"][I]['fields']['numer_sta'] \
                    for I in range(len(data_meteo["records"]))]
    list_station = np.unique(list_station).tolist()
    
    list_rec = []
    for sta_curr in list_station:
        list_rec_curr = []
        for rec_curr in data_meteo["records"]:
            if rec_curr['fields']['numer_sta'] == sta_curr:
                list_rec_curr.append(rec_curr)
        list_rec.append(list_rec_curr)
            
    return list_rec


def get_field_in_list(list_rec, field_name):
    '''
    get a field from a list of list of data 
    '''
    list_field = []
    for I in range(len(list_rec)):
        list_field_curr = []
        for J in range(len(list_rec[I])):
            try:
                list_field_curr.append(list_rec[I][J]["fields"][field_name])
            except:
                continue
        if list_field_curr != []:
            list_field.append(list_field_curr)
    return list_field

def calculate_mean_field(list_field, fun):
    """
    calculate mean with fun on list of data
    """
    list_by_sta = []
    for list_curr in list_field:
        list_by_sta.append(fun(list_curr))

    return np.mean(list_by_sta)

def calc_list_mean_field(data_meteo, fieldname, fun):
    list_date = get_data_meteo_date_list(data_meteo)
    list_mean = []
    for date_curr in list_date:
        data_out = get_rec_by_date(data_meteo, date_curr)
        list_by_sta = select_rec_by_station(data_out)
        list_field = get_field_in_list(list_by_sta, fieldname)
        list_mean.append(calculate_mean_field(list_field, fun))
    return list_mean

def get_data_pos():
    '''
    1) Retrieve data from Sante Publique France direct CSV URL 
        (updated every days but with 4 to 5 days delay...)
    2) Proceed this data by departements (tested - positive)
    3) Retrieve data from Méteo France
    4) Proceed this data to have mean feature all over France every days
    5) Proceed features data for model by combining all these data

    Every database is saved in CSV format.
    '''
    # patch 29/07/2020 : SSL error patch
    req = requests.get(URL_CSV_GOUV_FR).content
    df_gouv_fr_raw = pd.read_csv(io.StringIO(req.decode('utf-8')), sep=";")

    # past treat data upper cases -> lower cases
    if "t" not in df_gouv_fr_raw.columns:
        df_gouv_fr_raw["t"] =  df_gouv_fr_raw["T"]
    if "p" not in df_gouv_fr_raw.columns:
        df_gouv_fr_raw["p"] =  df_gouv_fr_raw["P"]
    # patch : clear data in double !!!
    df_gouv_fr_raw = df_gouv_fr_raw[df_gouv_fr_raw["cl_age90"] != 0]

    # creation of table data : 't':tested 'p':positive
    # data =  f(line : date, dep / col: t) => f(line : date / col: dep = f(t)) 
    pt_fr_test = pd.pivot_table(df_gouv_fr_raw, values=['t', 'p'], 
                                index=["jour"],
                        columns=["dep"], aggfunc=np.sum) 
    pt_fr_test["date"] = pt_fr_test.index

    # save data
    df_pos_fr = pt_fr_test["p"].copy()
    df_pos_fr.index = pt_fr_test["date"].index
    df_pos_fr["date"] = df_pos_fr.index
    df_pos_fr.to_csv(PATH_DF_POS_FR, index=False)

    df_test_fr = pt_fr_test["t"].copy()
    df_test_fr.index = pt_fr_test["date"].index
    df_test_fr["date"] = df_test_fr.index
    df_test_fr.to_csv(PATH_DF_TEST_FR, index=False)

    # meteo
    if os.path.isfile(PATH_JSON_METEO_FR):
        f_reload_from_start = False
        f_load_missing = False
        # load
        with open(PATH_JSON_METEO_FR) as f:
            data_meteo = json.load(f)
        # check start date
        date_meteo_start = get_data_meteo_date_min(data_meteo)
        delta_days = days_between(df_pos_fr.date.min(), date_meteo_start)
        if delta_days.days > 0:
            print(f"Must reload from start, {delta_days.days} days missing")
            f_reload_from_start = True
        # check last date
        date_meteo_end = get_data_meteo_date_max(data_meteo)
        delta_days = days_between(date_meteo_end, df_pos_fr.date.max())
        if delta_days.days > 0:
            print(f"Must load more last days, {delta_days.days} days missing")
            f_load_missing = True
        
        # determine list of days to download
        list_dates = None
        if f_reload_from_start:
            # all dates between [FORCED]
            list_dates = df_pos_fr["date"].tolist()
        elif f_load_missing:
            # from date
            list_dates = df_pos_fr["date"].tolist()
            # remove days already downloaded:
            list_remove = get_data_meteo_date_list(data_meteo)
            #get_data_meteo_date_list(data_meteo)
            for item_curr in  list_remove:
                try:
                    list_dates.remove(item_curr)
                except:
                    print(f'{item_curr} not found in dates list')
        else:
            # download NOT needed
            list_dates = None
    else:
        # all dates between [FORCED]
        f_reload_from_start = True
        f_load_missing = True
        list_dates = df_pos_fr["date"].tolist()
    # if download needed
    if list_dates is not None:
        data_meteo_new = get_data_meteo_by_list(list_dates)
        print(f'{len(data_meteo_new["records"])} records downloaded')

        if f_reload_from_start:
            # reload all
            data_meteo = data_meteo_new
        else:
            # add data
            data_meteo["records"] = data_meteo["records"] + \
                data_meteo_new["records"]      
        # save
        with open(PATH_JSON_METEO_FR, 'w') as outfile:
            json.dump(data_meteo, outfile)
    else:
        print("Data meteo OK")
    
    # create features for model

    # add meteo data

    list_t_min = calc_list_mean_field(data_meteo, "t", min)
    list_t_max = calc_list_mean_field(data_meteo, "t", max)
    list_u_min = calc_list_mean_field(data_meteo, "u", min)
    list_u_max = calc_list_mean_field(data_meteo, "u", max)

    dict_meteo = dict()
    dict_meteo["date"] = get_data_meteo_date_list(data_meteo)
    dict_meteo["t_min"] = list_t_min
    dict_meteo["t_max"] = list_t_max
    dict_meteo["u_min"] = list_u_min
    dict_meteo["u_max"] = list_u_max

    df_feat_fr = pd.DataFrame(data=dict_meteo)
    df_feat_fr.columns = ["date", "T_min", "T_max", "H_min", "H_max"]
    df_feat_fr.sort_values(by="date", inplace=True)
    df_feat_fr.index = df_feat_fr["date"]

    # add positive cases
    df_pos_fr["pos"] = 0
    list_dep = []
    for col_curr in df_pos_fr.columns:
        if re.search("^\d", col_curr):
            list_dep.append(col_curr)

    for dep_curr in list_dep:
        df_pos_fr["pos"] += df_pos_fr[dep_curr]

    df_feat_fr["pos"] = df_pos_fr["pos"].copy()

    # add tested cases
    df_test_fr["test"] = 0
    list_dep = []
    for col_curr in df_test_fr.columns:
        if re.search("^\d", col_curr):
            list_dep.append(col_curr)

    for dep_curr in list_dep:
        df_test_fr["test"] += df_test_fr[dep_curr]
        
    df_feat_fr["test"] = df_test_fr["test"].copy()

    # add num days
    df_feat_fr['day_num'] = \
        df_feat_fr["date"].astype(np.datetime64).dt.strftime("%w")

    # add nb_cases
    arr_nb_cases = df_feat_fr["pos"].cumsum().values
    df_feat_fr["nb_cases"] = NB_POS_DATE_MIN_DF_FEAT + arr_nb_cases
    df_feat_fr.to_csv(PATH_DF_FEAT_FR, index=False)



# FOR data to plot
def load_data_pos():
    '''
    Load data positive cases France
    '''
    df_feat_fr = pd.read_csv(PATH_DF_FEAT_FR)
    df_feat_fr.index = df_feat_fr["date"]
    return df_feat_fr

def update_pos(df_feat_fr):
    '''
    Update plot data positive cases France
    '''
    # pos last 60 days : date, pos, total (sum)
    str_date_0 = add_days(df_feat_fr.date.max(), -60)
    df_plot = df_feat_fr[df_feat_fr["date"] >= str_date_0].copy()
    return df_plot

def update_pred_pos(df_feat_fr):
    '''
    Update prediction data positive cases France
    '''
    # load model
    multi_step_model = tf.keras.models.load_model(PATH_MDL_MULTI_STEP)

    # prepare features
    features = df_feat_fr.copy().filter(items=['T_min', 'T_max', 'H_min',
                                            'H_max', 'pos', 'test', 'day_num'])
    # prepare dataset 
    dataset = features.values
    data_mean = dataset[:train_split].mean(axis=0)
    data_std = dataset[:train_split].std(axis=0)
    dataset = (dataset-data_mean)/data_std
    # prepare data : very last days
    x_multi = np.array([dataset[-past_history:,:]]) 
    # predict next days
    y_multi_pred = multi_step_model.predict(x_multi)
    # convert in positive cases
    y_pos_pred = y_multi_pred * data_std[4] + data_mean[4]
    # pos pred next 3 days from last day : date, pos, total (sum)
    str_date_pred_0 = df_feat_fr.date.max()
    str_date_pred_1 = add_days(str_date_pred_0, 3)
    list_dates_pred = generate_list_dates(str_date_pred_0, str_date_pred_1)
    # figure 
    df_plot_pred = pd.DataFrame(index=list_dates_pred, columns=["date"], 
                        data=list_dates_pred)

    df_plot_pred["pos"] = y_pos_pred[0].astype(int)
    arr_nb_pred = df_plot_pred["pos"].cumsum().values
    df_plot_pred["nb_cases"] = df_feat_fr["nb_cases"].max() + arr_nb_pred

    return df_plot_pred

def jsonifed_pred(df_plot_pred):
     return df_plot_pred.to_json(date_format='iso', orient='split')

# FOR FIGURE
def create_fig_pos(df_plot, df_plot_pred):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Create and style traces
    # total
    fig.add_trace(go.Scatter(x=df_plot["date"].astype(np.datetime64), 
                            y=df_plot["nb_cases"],
                        mode='lines+markers',
                        line_shape='linear',
                        connectgaps=True, name="Total cases"),
                secondary_y=False)

    fig.add_trace(go.Scatter(x=df_plot_pred["date"].astype(np.datetime64), 
                            y=df_plot_pred["nb_cases"],
                        mode='lines+markers',
                        line_shape='linear',
                        connectgaps=True, name="Total predicted"),
                secondary_y=False)
    # new cases
    fig.add_trace(go.Bar(x=df_plot["date"].astype(np.datetime64), 
                        y=df_plot["pos"], 
                        name="New cases"), 
                secondary_y=True)

    fig.add_trace(go.Bar(x=df_plot_pred["date"].astype(np.datetime64), 
                y=df_plot_pred["pos"], 
                name="New predicted"), 
                secondary_y=True)
    # Edit the layout
    fig.update_layout(title='COVID-19 Confirmed cases (France)',
                    yaxis_title='Total cases')
    fig.update_layout(legend_orientation="h", legend=dict(x=0, y=1.1))

    fig.update_yaxes(title_text="New cases", range=[0, 5000], secondary_y=True)

    return fig

def check_update():
    '''
    Just check if new data possibly available
    (if file date older than 24 hours)
    '''
    time_file_df_feat_date = get_file_date(PATH_DF_FEAT_FR)
    dtime_now  = datetime.datetime.now() - time_file_df_feat_date
    # update only if more than 24 hours without update
    if dtime_now.days > 0:
        flag_old = True
        # update data from external 
        print("Maybe new data available...")
        #get_data_pos()
    else:
        flag_old = False
        print("No new data available.")
    return flag_old

# APP DASH
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def startup_layout():
    '''
    startup web page
    '''
    #time_file_df_feat_date = get_file_date(PATH_DF_FEAT_FR)
    #dtime_now  = datetime.datetime.now() - time_file_df_feat_date

    # update 
    if check_update():
        get_data_pos()
    
    df_feat_fr = load_data_pos()
    
    df_plot = update_pos(df_feat_fr)
    df_plot_pred = update_pred_pos(df_feat_fr)

    str_date_mdl =  df_feat_fr.iloc[train_split]["date"]

    return html.Div(children=[
        html.H1(children='COVID-19 Cases Prediction in France'),
        html.Div(children='''
        LMST deep learning model : predict 3 next days with 10 last days
        '''),
        html.Div(children='Model trained until : {}'.format(str_date_mdl)),
        html.Button('Update Data', id='update-data', n_clicks=0),
        dcc.Loading(
            id="loading-fig-pos",
            type="default",
            children=html.Div(id="loading-output-1")
        ),
        dcc.Graph(id='covid-pos-graph',
            figure=create_fig_pos(df_plot, df_plot_pred)
        ),
        # Hidden div inside the app that stores the intermediate value
        html.Div(id='predicted-value', style={'display': 'none'},
            children=jsonifed_pred(df_plot_pred))
        ])

app.layout = startup_layout


# button update data
'''
@app.callback(
    dash.dependencies.Output('covid-pos-graph', 'figure'),
    [dash.dependencies.Input('update-data', 'n_clicks')])
def update_fig_pos(n_clicks):
    df_feat_fr = load_data_pos()
    df_plot = update_pos(df_feat_fr)
    df_plot_pred = update_pred_pos(df_feat_fr)
    return create_fig_pos(df_plot, df_plot_pred)
'''

@app.callback(
    [dash.dependencies.Output('loading-output-1', 'children') , 
    dash.dependencies.Output('covid-pos-graph', 'figure')],
    [dash.dependencies.Input('update-data', 'n_clicks')],
    [dash.dependencies.State('predicted-value', 'children')])
def load_figure(n_clicks, jsonified_pred):
    flag_update = check_update()
    if flag_update:
        get_data_pos()
    df_feat_fr = load_data_pos()
    df_plot = update_pos(df_feat_fr)
    
    if flag_update:
        # model predicting
        df_plot_pred = update_pred_pos(df_feat_fr)
    else:
        # load from hidden div (no model predicting again)
        print("loading prediction from hidden div...")
        df_plot_pred = pd.read_json(jsonified_pred, orient='split')
    
    return conv_dt_2_str(get_file_date(PATH_DF_FEAT_FR)), \
                        create_fig_pos(df_plot, df_plot_pred)

if __name__ == '__main__':
    app.run_server(debug=True)
    app.config.suppress_callback_exceptions = True

