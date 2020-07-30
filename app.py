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

# DEFINITIONS
PATH_TO_SAVE_DATA = "."
PATH_DF_POS_FR = PATH_TO_SAVE_DATA + '/' + 'df_pos_fr.csv'
PATH_DF_TEST_FR = PATH_TO_SAVE_DATA + '/' + 'df_test_fr.csv'
PATH_JSON_METEO_FR = PATH_TO_SAVE_DATA + '/' + 'data_meteo_fr.json'
PATH_DF_FEAT_FR = PATH_TO_SAVE_DATA + '/' + 'df_feat_fr.csv' 
PATH_GEO_DEP_FR = PATH_TO_SAVE_DATA + '/sources/geofrance/' + 'departments.csv'
PATH_MDL_SINGLE_STEP = PATH_TO_SAVE_DATA + '/' + "mdl_single_step_pos_fr"
PATH_MDL_MULTI_STEP = PATH_TO_SAVE_DATA + '/' + "mdl_multi_step_pos_fr"

date_format = "%Y-%m-%d"

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

# EXTRACT DATA
def extract_data_world(df_world_melt, df_death_melt, str_filter, str_value):
    '''
    Extract data cases & death for one country by date from df_world_melt
    
    '''

    df_cases = df_world_melt[df_world_melt[str_filter] == str_value]
    

    s_cases = df_cases.groupby("date")["nb_cases"].sum()
    df_cases_out = pd.DataFrame(columns=["date", "nb_cases"])
    df_cases_out["nb_cases"] = s_cases.values
    df_cases_out["date"] = s_cases.index
    
    if df_death_melt is not None:
        df_death = df_death_melt[df_death_melt[str_filter] == str_value]
        s_death = df_death.groupby("date")["nb_death"].sum()
        df_death_out = pd.DataFrame(columns=["date", "nb_death"])
        df_death_out["nb_death"] = s_death.values
        df_death_out["date"] = s_death.index
        return df_cases_out, df_death_out 
    else:
        print("one output ...")
        return df_cases_out 


# load model
multi_step_model = tf.keras.models.load_model(PATH_MDL_MULTI_STEP)
# model parameters
train_split = 50
past_history = 10 # days used to predict next values in future
future_target = 3 # predict 3 days later
STEP = 1

# load data
# reload data
df_feat_fr = pd.read_csv(PATH_DF_FEAT_FR)
df_feat_fr.index = df_feat_fr["date"]
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

###############
# FROM JHU CSSE
#
# confirmed cases
PATH_WORLD_CONF = 'https://raw.githubusercontent.com/CSSEGISandData/' + \
    'COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/' + \
    'time_series_covid19_confirmed_global.csv'
df_world = pd.read_csv(PATH_WORLD_CONF)

# fix problem no data at all for one date
df_world.dropna(axis='columns', how='all', inplace=True)

df_world_melt = df_world.melt(id_vars=["Province/State", "Country/Region", 
                                       "Lat", "Long"],
                            value_vars=df_world.columns[4:], 
                            var_name="date", value_name="nb_cases")

df_world_melt["Province/State"] = df_world_melt["Province/State"].fillna(" ")

df_world_melt["area"] = df_world_melt["Country/Region"] + " : " +\
    df_world_melt["Province/State"]  

df_world_melt["date"] = df_world_melt["date"].astype(np.datetime64)

df_world_melt.sort_values(by=['date'], inplace=True)

df_world_melt["nb_cases"] = df_world_melt["nb_cases"].fillna(0)
# path because sometimes, value are negative !
df_world_melt["nb_cases"] = df_world_melt["nb_cases"].apply(math.fabs)

# remove USA
df_world_melt = df_world_melt[df_world_melt["Country/Region"] != 'US']


df_cases_fr = extract_data_world(df_world_melt, None,
                                              "Country/Region", "France")

# pos last 28 days : date, pos, total (sum)
str_date_0 = add_days(df_feat_fr.date.max(), -60)
df_plot = df_feat_fr[df_feat_fr["date"] >= str_date_0].copy()

nb_0 = df_cases_fr[df_cases_fr["date"] == str_date_0]['nb_cases'].values[0]
arr_nb = df_feat_fr[df_feat_fr["date"] >= str_date_0]["pos"].cumsum().values
df_plot["nb_cases"] = nb_0 + arr_nb

# pos pred next 3 days from last day : date, pos, total (sum)
str_date_pred_0 = df_feat_fr.date.max()
str_date_pred_1 = add_days(str_date_pred_0, 3)
list_dates_pred = generate_list_dates(str_date_pred_0, str_date_pred_1)
# figure 
df_plot_pred = pd.DataFrame(index=list_dates_pred, columns=["date"], 
                       data=list_dates_pred)

df_plot_pred["pos"] = y_pos_pred[0].astype(int)
arr_nb_pred = df_plot_pred["pos"].cumsum().values
df_plot_pred["nb_cases"] = df_plot["nb_cases"].max() + arr_nb_pred

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
                   yaxis_title='nb confirmed cases')
fig.update_layout(legend_orientation="h", legend=dict(x=0, y=1.1))

fig.update_yaxes(range=[0, 5000], secondary_y=True)

# APP DASH
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='COVID-19 Cases Prediction in France'),

    html.Div(children='''
        LMST deep learning model : predict 3 next days with 10 last days
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
