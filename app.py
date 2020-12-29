# -*- coding: utf-8 -*-

''' App Covid Visu web app
COVID-19 in France Dashboard: Datavisualization & Model
Map of Reproduction number in France by "départements"
A tensorflow deep leanring model try to estimate next 7 days confirmed cases in 
France with last 14 days data available.

Run this app with `python app.py` and
visit http://0.0.0.0/ in your web browser.
'''
# IMPORT 

# import bluit-in 
import math
import datetime
import re
import os
import sys
# import third party 
#import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input
from dash.dependencies import Output
from dash.dependencies import State
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import project modules 
import settings
from my_helpers.dates import *
from my_helpers.data_plots import prepare_data_input
from my_helpers.data_plots import prepare_plot_data_pos
from my_helpers.data_plots import check_update
from my_helpers.data_plots import PATH_DF_POS_FR
from my_helpers.data_plots import load_data_pos
from my_helpers.data_plots_kr import load_df_feat_kr
from my_helpers.data_plots_kr import GEOJSON_KR
from my_helpers.data_plots_kr import LIST_NAME_GEOJSON
from my_helpers.data_plots_kr import LAT_LON_KR
from my_helpers.data_plots_kr import ZOOM_KR
from my_helpers.data_plots_kr import LIST_SUM_GEOJSON
from my_helpers.data_plots_kr import LIST_RT_GEOJSON
from my_helpers.data_plots_kr import DICT_AREA
from my_helpers.data_plots_kr import update_df_feat_kr
from my_helpers.model import calc_rt
from my_helpers.model import NB_DAYS_CV
from my_helpers.model import FUTURE_TARGET, PAST_HISTORY
from my_helpers.data_maps import prepare_plot_data_map
from my_helpers.utils import sum_mobile
# DEFINITIONS

PATH_TO_SAVE_DATA = settings.PATH_TO_SAVE_DATA

meta_tags=[{
      'name': 'viewport',
      'content': 'width=device-width, initial-scale=1.0'
    }
]

# HELPER FUNCTIONS

def display_msg(my_message):
    print("{} : {}".format(\
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), my_message))

def jsonifed_pred(df_plot_pred):
    '''Tranform DataFrame into json with date '''
    return df_plot_pred.to_json(date_format='iso', orient='split')

# FIGURE FUNC

def create_fig_pos(df_plot, df_plot_pred, df_plot_pred_all, str_date_mdl):
    display_msg("create_fig_pos...")
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Create and style traces
    # total
    fig.add_trace(go.Scatter(x=df_plot["date"].astype(np.datetime64), 
                            y=df_plot["nb_cases"],
                        mode='lines+markers',
                        line_shape='linear',
                        line_color="blue",
                        connectgaps=True, name="Total"),
                secondary_y=False)
    fig.add_trace(go.Scatter(x=df_plot_pred_all["date"].astype(np.datetime64), 
                            y=df_plot_pred_all["nb_cases"],
                        mode='lines+markers',
                        line_shape='hv',
                        line_color="red",
                        connectgaps=True, name="Total (estim.)"),
                secondary_y=False)
    fig.add_trace(go.Scatter(x=df_plot_pred["date"].astype(np.datetime64), 
                            y=df_plot_pred["nb_cases"],
                        mode='lines+markers',
                        line_shape='hv',
                        line_color="orange",
                        connectgaps=True, name="Total (future estim.)"),
                secondary_y=False)
    # new cases
    fig.add_trace(go.Bar(x=df_plot["date"].astype(np.datetime64), 
                        y=df_plot["pos"], 
                        name="Daily", opacity=0.33, marker_color="blue"), 
                secondary_y=True)
    fig.add_trace(go.Bar(x=df_plot_pred["date"].astype(np.datetime64), 
            y=df_plot_pred["pos"], 
            name="Daily (future estim.)", opacity=0.33, marker_color="orange"), 
                secondary_y=True)
    fig.add_trace(go.Scatter(x=df_plot_pred_all["date"].astype(np.datetime64), 
                            y=df_plot_pred_all["pos"],
                        mode='lines+markers',
                        marker_symbol="cross",
                        line_color="red", opacity=0.33,    
                        connectgaps=True, name="Daily (estim.)"),
                secondary_y=True)

    # Edit the layout
    title_fig = '<b>COVID-19 Confirmed cases in France</b>' + \
        '<br>Model trained until <b>' + str_date_mdl + '</b>' + \
        '<br>predicts next {} days with last {} days until <b>' \
        .format(FUTURE_TARGET, PAST_HISTORY) + \
        df_plot_pred["date"].max() + '</b>'
    fig.update_layout(title=title_fig, yaxis_title='nb <b>Total</b> cases')
    fig.update_layout(legend_orientation="h", legend=dict(x=0, y=1))
    fig.update_layout(height=600)

    fig.update_yaxes(title_text="nb <b>Daily</b> cases", secondary_y=True)
    display_msg("create_fig_pos END")
    return fig

def create_fig_rt(df_dep_r0, df_code_dep, pt_fr_test_last):
    display_msg("create_fig_rt...")
    list_num_dep = df_dep_r0.columns[1:].tolist()
    # path 975 & 977 & 978 doesn't exist in dep name data
    list_num_dep.remove('975')
    list_num_dep.remove('977')
    list_num_dep.remove('978')

    list_name_dep = [f'{dep_num_curr} - ' + \
                df_code_dep.loc[df_code_dep["code"] == dep_num_curr,
                                    "name"].values[0] + \
                "<br>Rt=<b>{:.2f}</b>".format(df_dep_r0[dep_num_curr][-1]) + \
                " cases={}".format(pt_fr_test_last.loc[ \
                pt_fr_test_last.dep == dep_num_curr, "p"].values[0]) \
                for dep_num_curr in list_num_dep]

    nb_dep = len(list_num_dep)
    nb_col = 4
    nb_row = math.ceil(nb_dep/nb_col)
    fig = make_subplots(rows=nb_row, cols=nb_col, shared_xaxes=True,
                        shared_yaxes=True,
                        subplot_titles=list_name_dep)
    I_dep = 0
    #list_color = []
    for row in range(nb_row):
        for col in range(nb_col):   
            dep_num_curr = list_num_dep[I_dep]
            dep_curr = df_code_dep.loc[df_code_dep["code"] == dep_num_curr, 
                                    "name"].values[0]
        
            if (df_dep_r0[dep_num_curr][-1] > 1) & \
                (pt_fr_test_last.loc[ \
                    pt_fr_test_last.dep == dep_num_curr, "p"].values[0] > 400):
                color_curr = "red"
            elif (df_dep_r0[dep_num_curr][-1] > 1):
                color_curr = "orange"
            else:
                color_curr = "blue"
        
            
            fig.add_trace(go.Scatter(x=df_dep_r0["date"], 
                        y=df_dep_r0[dep_num_curr],
                        mode='lines', name=dep_curr, 
                        line=dict(color=color_curr),
                        fill='tozeroy'), 
                        row=row+1, col=col+1)
            
            fig.add_trace(go.Scatter(x=[df_dep_r0["date"][0], 
                                        df_dep_r0["date"][-1]], 
                                    y=[1,1],
                                    mode='lines', 
                                    line=dict(color="red", dash='dash'),
                                    hoverinfo="skip"), 
                        row=row+1, col=col+1)
            I_dep +=1
            
            if I_dep >= nb_dep: #nb_dep:
                break
        if I_dep >= nb_dep: #nb_dep:
                break 

    #fig.update_traces(patch=dict(font=dict(size=6)))
    for I, subplot_title_curr in enumerate(fig['layout']['annotations']):
        subplot_title_curr['font'] = dict(size=10)
        subplot_title_curr['xshift'] = 0
        subplot_title_curr['yshift'] = -10

    fig.update_layout(
        height=1800,
        title="Rt: Estimated Reproduction Nb. in France ( until {} )".format( \
            df_dep_r0['date'].max()),
        showlegend=False,
        font=dict(
            size=12,
        )
    )
    display_msg("create_fig_rt END.")
    return fig

def create_fig_map(pt_fr_test_last, dep_fr, str_date_last):
    '''Graph Rt map France
    figure map of confirmed / testers and reproduction number by "départements"
     data : 
     - dep_fr (geo json )
     - pt_fr_test_last : pivot table : sum up last 14 days of confirmed cases
    '''
    display_msg("create_fig_map...")
    lat_lon_fr =  {'lat':  47, 'lon': 2}
    zoom_fr = 4.25
    mapbox_args_fr = {'center': lat_lon_fr, 
                    'style': 'carto-positron', 'zoom': zoom_fr}

    mapbox_args_idf = {'center': {'lat':  48.86, 'lon': 2.33}, 
                    'style': 'carto-positron', 'zoom': 7}
    mapbox_args_dom = {'center': {'lat':  17, 'lon': -2}, 
                    'style': 'carto-positron', 'zoom': 2}

    # Initialize figure

    fig = go.Figure()

    # Add Traces

    fig.add_trace(
        go.Choroplethmapbox(geojson=dep_fr, name="positive",
                                    locations=pt_fr_test_last["name"], 
                                    featureidkey="properties.nom",
                                    z=pt_fr_test_last["p"],
                                    marker_opacity=0.7, marker_line_width=0))

    fig.add_trace(
        go.Choroplethmapbox(geojson=dep_fr, name="tested",
                                    locations=pt_fr_test_last["name"], 
                                    featureidkey="properties.nom",
                                    z=pt_fr_test_last["t"],
                                    marker_opacity=0.7, marker_line_width=0,
                                    visible=False))

    fig.add_trace(
        go.Choroplethmapbox(geojson=dep_fr, name="Rt",
                                    locations=pt_fr_test_last["name"], 
                                    featureidkey="properties.nom",
                                    z=pt_fr_test_last["R0"], zmin=.5, zmax=1.5,
                                    marker_opacity=0.7, marker_line_width=0,
                                    visible=False))

    annot_conf=[dict( \
        text="France : <b>Confirmed</b> cases (Total for 14 days before " + \
        f"{str_date_last})", 
                    x=0, xref="paper", y=1, yref="paper",
                                align="left", showarrow=False,
                    bgcolor="#FFFFFF")]

    annot_test=[dict( \
        text="France : <b>Tested</b> cases (Total for 14 days before " + \
        f"{str_date_last})", x=0, xref="paper", y=1, yref="paper",
                                align="left", showarrow=False,
                    bgcolor="#FFFFFF")]

    annot_r0=[dict( \
        text="France : <b>Rt</b> estimated for 14 days before " + \
        f"{str_date_last})", x=0, xref="paper", y=1, yref="paper",
                                align="left", showarrow=False,
                    bgcolor="#FFFFFF")]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                xanchor="left",
                y=0.95,
                x=0,
                active=0,
                showactive=True,
                buttons=list([
                    dict(label="Confirmed",
                        method="update",
                        args=[{"visible": [True, False, False]},
                            {"annotations": annot_conf}]),
                    
                    dict(label="Tested",
                        method="update",
                        args=[{"visible": [False, True, False]},
                            {"annotations": annot_test}]),
                    
                    dict(label="Rt",
                        method="update",
                        args=[{"visible": [False, False, True]},
                            {"annotations": annot_r0}]), 
                    
                    dict(label="Zoom : IdF",
                        method="relayout",
                        args=[{"mapbox": mapbox_args_idf}]),
                    
                dict(label="France",
                        method="relayout",
                        args=[{"mapbox": mapbox_args_fr}]), 
                    
                dict(label="DOM-TOM",
                        method="relayout",
                        args=[{"mapbox": mapbox_args_dom}]),
                ]),
            )
        ])

    fig.update_layout(mapbox_style="carto-positron",
                    mapbox_zoom=zoom_fr, mapbox_center = lat_lon_fr)

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(annotations=annot_conf)

    fig.update_traces(colorbar=dict(thicknessmode="pixels", thickness=10,
        len=0.8,
        x=0.9,
        xanchor="left",
        xpad=0),
        selector=dict(type='choroplethmapbox'))

    display_msg("create_fig_map END.")
    return fig

def create_fig_map_kr(df_feat_kr, list_col, label):
    '''Graph Rt map France
    figure map of confirmed / testers and reproduction number by "départements"
     data : 
     - dep_fr (geo json )
     - pt_fr_test_last : pivot table : sum up last 14 days of confirmed cases
    '''
    display_msg("create_fig_map_kr...")
    
    str_date_last = df_feat_kr.index[-1].strftime("%Y-%m-%d")

    # Initialize figure
    fig = go.Figure()

    # Add Traces
    fig.add_trace(
        go.Choroplethmapbox(geojson=GEOJSON_KR, name="positive",
                    locations=LIST_NAME_GEOJSON, 
                    featureidkey="properties.NAME_1",
                    z=df_feat_kr.filter(list_col).iloc[-1].values,
                    marker_opacity=0.7, marker_line_width=0))

    annot_conf=[dict( \
        text="South Korea : " + label + f" (up to {str_date_last})", 
                    x=0, xref="paper", y=1, yref="paper",
                                align="left", showarrow=False,
                    bgcolor="#FFFFFF")]


    fig.update_layout(mapbox_style="carto-positron",
                    mapbox_zoom=ZOOM_KR, mapbox_center = LAT_LON_KR)

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(annotations=annot_conf)

    fig.update_traces(colorbar=dict(thicknessmode="pixels", thickness=10,
        len=0.8,
        x=0.9,
        xanchor="left",
        xpad=0),
        selector=dict(type='choroplethmapbox'))

    display_msg("create_fig_map_kr END.")
    return fig


def figure_rt(ser_rt, dep_curr, sum_pos_last, country="France"):

    # color calculation
    if (ser_rt.values[-1] > 1) & \
    (sum_pos_last > 400):
        color_curr = "red"
    elif (ser_rt.values[-1] > 1):
        color_curr = "orange"
    else:
        color_curr = "blue"
        
    # create figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ser_rt.index, y=ser_rt.values,
                mode='lines', name=dep_curr, line=dict(color=color_curr),
                fill='tozeroy'))

    fig.add_trace(go.Scatter(x=[ser_rt.index[0], 
                                    ser_rt.index[-1]], 
                                y=[1,1],
                                mode='lines', 
                                line=dict(color="red", dash='dash'),
                                hoverinfo="skip"))

    fig.add_annotation(
                x=0,
                y=-0.18,
                text="<i>Click on Map to Update this Curve<br> " + \
                    f'or Click on "{country}" ' + \
                    "button for global country Curve</i>")
    fig.update_annotations(dict(
                xref="paper",
                yref="paper",
                showarrow=False
    ))
    str_date = ser_rt.index[-1]
    if isinstance(str_date, pd.Timestamp):
        str_date = str_date.strftime("%Y-%m-%d")
    subtitle_curr = "Rt: " + \
                    "<b>{:.2f}</b> ".format(ser_rt.values[-1]) + \
                    f'on {str_date}<br>' + \
                    f"sum cases: <b>{sum_pos_last}</b> (last 14 days)"

    fig.update_layout(
        title=dict(text="<b>Reprod. nb.</b>: <b>{}</b>".format(dep_curr) + \
            '<br>' + subtitle_curr, 
            xanchor="center", x=0.5, yanchor="top", y=0.95),
        yaxis_title='Rt',
        showlegend=False,
        font=dict(
            size=12,
        )
    )

    fig.update_layout(margin={"r":0,"t":70, "l":50}) 

    fig.update_yaxes(title_standoff=0)
    return fig

def create_fig_rt_dep(dep_curr, df_code_dep, pt_fr_test_last, df_dep_r0):
    
    '''Rt evolution plots for one departement
    
     data : 
     - df_dep_r0 (date,  date / dep. Rt)
     - df_code_dep (-, dep. code / dep name )
     - pt_fr_test_last (-, p / t / dep / code / name / p_0 / R0)
    '''
    display_msg("create_fig_rt_dep ...")
    dep_num_curr = df_code_dep.loc[df_code_dep["name"] == dep_curr, 
                            "code"].values[0]


    ser_rt = df_dep_r0[dep_num_curr]
    sum_pos_last = pt_fr_test_last.loc[ \
                        pt_fr_test_last.dep == dep_num_curr, "p"].values[0]


    display_msg("create_fig_rt_dep END.")
    return figure_rt(ser_rt, dep_curr, sum_pos_last)

def create_fig_rt_fr(df_feat_fr):
    
    '''Rt evolution plots for france
    
     data : 
     - df_feat_fr (date,  [date, pos] )
    '''
    display_msg("create_fig_rt_fr ...")

    ser_rt = df_feat_fr["Rt"]
    sum_last = df_feat_fr["sum_cases"].values[-1]

    display_msg("create_fig_rt_fr END.")

    return figure_rt(ser_rt, "France", sum_last)

def figure_pos(ser_pos, ser_sum_pos, dep_curr, rt_last):
    '''
    Figure creation positive daily and sum-mobile
    '''
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ser_pos.index, y=ser_pos.values,
                mode='lines', name="daily", line=dict(color="red"),
                fill='tozeroy'), secondary_y=False)

    fig.add_trace(go.Scatter(x=ser_sum_pos.index, y=ser_sum_pos.values,
                mode='lines', name='14-days-sum', 
                line=dict(color="blue")), secondary_y=True)

    fig.add_annotation(
                x=0,
                y=-0.18,
                text="<i>Click on Map to Update this Curve<br>" + \
                    "Curve for global country not available...</i>")
    fig.update_annotations(dict(
                xref="paper",
                yref="paper",
                showarrow=False
    ))

    str_date = ser_pos.index[-1]
    if isinstance(str_date, pd.Timestamp):
        str_date = ser_pos.index[-1].strftime("%Y-%m-%d")

    subtitle_curr = \
        f'<i>{str_date}:</i> ' + \
        'Rt: <b>{:.2f}</b>'.format(rt_last) + \
        "<br>14days-sum:<b> {:.0f}</b>".format(ser_sum_pos.values[-1])

    fig.update_layout(showlegend=True, font=dict(size=12),
        title=dict(text=f"New cases: <b>{dep_curr}</b><br>" + \
        subtitle_curr,
        xanchor="center", x=0.5, yanchor="top", y=0.95)
    )

    fig.update_yaxes({"color": "red",}, secondary_y=False)

    fig.update_yaxes({"color": "blue"}, secondary_y=True) 
    fig.update_layout(margin={"r":0,"t":70, "l":50}) 
    fig.update_layout(legend_orientation="h", legend=dict(x=0, y=1))
    
    return fig

def create_fig_pos_dep(dep_curr, df_code_dep, df_dep_r0, 
        df_pos_fr, df_dep_sum):
    
    '''Confirmed evolution plots for one departement
    
     data : 
     - dep_curr : dept.name current (dept that you display)
     - df_dep_r0 (date, [ date  [Rt by dep.] ])
     - df_code_dep (-, [ dept.code, dept.name ])
     - pt_fr_test_last (-, 
        [ p, t,  dept.code,  dept.name,  cases,  R0 ]) for last 14 days
     - df_pos_fr (date, [ date, [daily cases by dep.] ])
    '''
    display_msg("create_fig_pos_dep ...")
    # choice dept.
    dep_num_curr = df_code_dep.loc[df_code_dep["name"] == dep_curr, 
                            "code"].values[0]
    # calculation
    pos_mean = df_dep_sum[dep_num_curr]

    # new input style (temp)
    rt_last = df_dep_r0[dep_num_curr][-1]
    
    display_msg("create_fig_rt_dep END.")

    return figure_pos(df_pos_fr[dep_num_curr], pos_mean, dep_curr, rt_last)


def create_fig_pos_rate(df_feat_fr, country="France"):
    '''
    data : 
     - df_feat_fr (date,  [date, pos , test, age_pos] )

    pos_rate =  100*df_feat_fr["pos"] / df_feat_fr["test"]

    '''
    display_msg("create_fig_pos_rate ...")

    rate_pos = df_feat_fr["rate_pos"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=df_feat_fr["date"], 
        y=rate_pos.values,
        mode='lines', name="pos. rate", line=dict(color="red"),
        fill='tozeroy'), secondary_y=False)

    fig.add_trace(go.Scatter(x=df_feat_fr["date"], 
            y=df_feat_fr["age_pos"],
            mode='lines', name='pos. age', 
            line=dict(color="blue")), secondary_y=True)
    
    age_last = df_feat_fr[df_feat_fr["age_pos"].notna()]["age_pos"].values[-1]
    
    subtitle_curr = \
        f'<i>{df_feat_fr["date"].values[-1]}:</i> ' + \
        'pos. rate:<b> {:.1f}</b>'.format(rate_pos.values[-1]) + \
        " %<br>mean pos. age:<b> {:.1f}</b>".format(age_last)

    fig.update_layout(showlegend=True, font=dict(size=12),
        title=dict(text=f"Positive rate and age: <b>{country}</b><br>" + \
        subtitle_curr,
        xanchor="center", x=0.5, yanchor="top", y=0.95)
    )

    fig.update_yaxes({"color": "red",}, secondary_y=False)

    fig.update_yaxes({"color": "blue"}, secondary_y=True)
    fig.update_layout(margin={"r":0,"t":70, "l":50}) 
    fig.update_layout(legend_orientation="h", legend=dict(x=0, y=1))

    fig.add_annotation(
                x=0,
                y=-0.18,
                text="<i>Only global country Curve available<br></i>")
    fig.update_annotations(dict(
                xref="paper",
                yref="paper",
                showarrow=False
    ))
    display_msg("create_fig_pos_rate END.")

    return fig


def get_title(str_country):
    return f'COVID-19 in {str_country} Dashboard: ' + \
            'Datavisualization & Model'

# APP DASH
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, 
    meta_tags=meta_tags)
app.title = "App Covid Visu"
server = app.server

def startup_layout():
    '''
    startup web page
    '''
    display_msg("STARTUP...")
    
    #time_file_df_feat_date = get_file_date(PATH_DF_FEAT_FR)
    #dtime_now  = datetime.datetime.now() - time_file_df_feat_date

    # update 
    if settings.MODE_FORCE_UPDATE:
        flag_update = True
    else:
        flag_update = check_update()
    
    # prepare input data
    df_feat_fr, str_date_mdl, str_data_date = prepare_data_input(flag_update)
    
    # prepare plot data for positive cases
    df_plot, df_plot_pred, df_plot_pred_all, str_date_last = \
        prepare_plot_data_pos(df_feat_fr, flag_update)

    # prepare plot data for MAPS
    df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep, df_dep_sum = \
        prepare_plot_data_map(flag_update)
    
    df_pos_fr = pd.read_csv(PATH_DF_POS_FR)
    df_pos_fr.index = df_pos_fr["date"]
    
    # update data KR
    df_feat_kr = update_df_feat_kr()
    str_date_kr = f'(up to: {df_feat_kr["date"].values[-1]})'

    # informations
    markdown_info = '''
    ***Legend***    
    `Total`                 : Actual total number of confirmed cases in France for past days  
    `Total (estim.)`        : Estimated total number of confirmed cases in France for past days (by model)  
    `Total (future estim.)` : Estimated total number of confirmed cases in France for future days (by model)  
    `Daily`                 : Actual daily number of confirmed cases in France for past days  
    `Daily (future estim.)` : Estimated daily number of confirmed cases in France for future days (by model)  
    `Daily (estim.)`        : Estimated daily number of confirmed cases in France for past days (by model)  
    
    ***About Model***  
      
    The model is a simple LSTM Deep Learning Tensorflow model.  
      
    It estimates the number of daily confirmed cases in France for next days by time-series forecast.  
      
    For that, the model takes a period of 14 days to estimate the next 7 days.  
      
    Because of lack of data, it has been trained with only few past periods and validated on only very few periods!  
      
    Input Features are daily data for:
    - Min/Max Temperatures
    - Min/Max Humidities
    - Confirmed cases
    - Tested cases
    - Day of the week
    - Mean Age of Tested cases
    - Mean Age of Confirmed cases

    The predictions are under-estimated because the evolution is big during last days.  
      
    The model will learn from this current changing period in few weeks, and it will be better.  
      
    If new data is available, the model is predicting daily confirmed cases for next days.  
      
    More info in my github below.  
      
    ***GitHub***: https://github.com/jeugregg/coronavirusModel
      
    ***DATA sources***: 
    - Tested / Confirmed cases: https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-resultats-des-tests-virologiques-covid-19
    - Météo France : https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm

    '''
    
    str_country = "France"

    
    display_msg("STARTUP END.")
    
    return html.Div(children=[
        html.H1(children=get_title(str_country), id="app-title"),
        html.Div(children=dcc.Dropdown(
            id='country-dropdown',
            options=[
                {'label': 'France', 'value': 'France'},
                {'label': 'South Korea', 'value': 'South Korea', 
                "disabled": False, "title": "Coming soon..."}
            ],
            value=str_country, clearable=False, searchable=False,
            ), style={'float': 'left', 'width': 120, 'margin-right': 10}),
        html.Div(children=html.Button('Update', id='update-data', 
        n_clicks=0), style={'display': 'inline-block', 'margin-right': 10}),
        html.Div(children=dcc.Loading(
            id="loading-fig-pos",
            type="default",
            children=[html.Div(id="loading-output-1", children=str_data_date), 
                html.Div(id="loading-output-kr", children=str_date_kr, 
                className="app-hide")]), 
            style={'display': 'inline-block', 'margin-right': 10}),
        html.Div(children=html.A(
            children="By Gregory LANG, Data Scientist Freelance",
            href="https://greg.coolplace.fr/data-scientist-freelance", 
            target="_blank"), style={'display': 'inline-block'}),

        dcc.Tabs(id='tabs-app', value='tab-mdl', children=[
            dcc.Tab(id="tab-mdl", 
                label='Evolution & Model', 
                value='tab-mdl', 
                children=[
                    dcc.Graph(id='covid-pos-graph',
                        figure=create_fig_pos(df_plot, df_plot_pred, 
                            df_plot_pred_all, str_date_mdl), 
                        className="graph-mdl"
                    ),
                    dcc.Graph(id='covid-pos-graph-kr', className="app-hide")
                ]
            ),
            dcc.Tab(id="tab-map",
                label='Maps', 
                value='tab-map', children=[
                html.Div(id="div-rt-map", 
                    children=[
                        dcc.Graph(id='covid-rt-map',
                            figure=create_fig_map(pt_fr_test_last, dep_fr, 
                            str_date_last), 
                            )], 
                    n_clicks=0, 
                    className="app-map"
                ),
                html.Div(id="div-rt-map-kr", 
                    children=[
                        html.Button('Confirmed', id='btn-conf', n_clicks=0,
                            style={"padding-right": 15, "padding-left": 15}),
                        html.Button('Tested', id='btn-test', n_clicks=0,
                            style={"padding-right": 15, "padding-left": 15}),
                        html.Button('Rt', id='btn-rt', n_clicks=0, 
                            style={"padding-right": 15, "padding-left": 15}),
                        html.Button('South Korea', id='btn-kr', n_clicks=0,
                            style={"padding-right": 15, "padding-left": 15}),
                        dcc.Graph(id='covid-rt-map-kr')
                    ], 
                    n_clicks=0, 
                    className="app-hide"
                ),
                html.Div(id="div-rt-curve", 
                    children=[
                        dcc.Loading(id="loading-graph-map", type="default"),
                        dcc.Graph(id='covid-rt-dep-graph', 
                            figure=create_fig_pos_dep("Paris", df_code_dep, 
                                df_dep_r0, df_pos_fr, df_dep_sum)
                        )
                    ], 
                    className="app-graph-map"
                ),
                html.Div(id="div-rt-curve-kr", 
                    children=[
                        dcc.Loading(id="loading-graph-map-kr", type="default"),
                        dcc.Graph(id='covid-rt-dep-graph-kr')
                    ], 
                    className="app-hide"
                )

            ])
            
        ], style={'margin-top': 10}),
        # Hidden div inside the app that stores the intermediate value
        html.Div(id='predicted-value', style={'display': 'none'},
            children=jsonifed_pred(df_plot_pred)),
        html.Div(id='predicted-value-all', style={'display': 'none'},
            children=jsonifed_pred(df_plot_pred_all)),
        html.Div(id='graph_type', style={'display': 'none'},
            children=0),
        html.Div(id='graph_type_kr', style={'display': 'none'},
            children=0),
        html.Div(id='id_button', style={'display': 'none'},
            children=0), 
        html.Div(id='mode_country', style={'display': 'none'},
            children=0),
        html.Div(id='dep', style={'display': 'none'},
            children=""),    
        html.Div(id='info', children=dcc.Markdown(children=markdown_info))
        ])

app.layout = startup_layout

@app.callback(
    [dash.dependencies.Output("app-title", "children"),
    dash.dependencies.Output('covid-pos-graph', 'className'),
    dash.dependencies.Output("div-rt-map", "className"),
    dash.dependencies.Output("div-rt-curve", "className"),
    dash.dependencies.Output("div-rt-map-kr", "className"),
    dash.dependencies.Output("div-rt-curve-kr", "className"),
    Output("loading-output-1", "className"),
    Output("loading-output-kr", "className")],
    [dash.dependencies.Input('country-dropdown', 'value')],
    prevent_initial_call=True)
def update_tabs(value):
    display_msg("update_tabs ...")
    if (value == "South Korea"):

        '''# load data kr
        df_feat_kr = load_df_feat_kr()
        # figure mdl
        graph_mdl = [""]
        # figure map
        fig_map = create_fig_map_kr(df_feat_kr, LIST_SUM_GEOJSON, 
            "<b>Confirmed</b> cases : Sum of last 14 days")


        # figure graph
        dep_curr = "South Korea"
        ser_pos = df_feat_kr["pos"]
        ser_sum_pos = df_feat_kr["sum_cases"]
        rt_last = df_feat_kr["Rt"].values[-1]
        fig_curve = figure_pos(ser_pos, ser_sum_pos, dep_curr, rt_last)
        graph_curve = dcc.Graph(id='covid-rt-map-curve',
                    figure=fig_curve)'''

        
        return get_title(value), "app-hide", "app-hide", "app-hide", \
            "app-map", "app-graph-map", "app-hide", "app-show-block"
    else:
        '''# prepare input data
        df_feat_fr, str_date_mdl, str_data_date = prepare_data_input(False)
    
        # prepare plot data for positive cases
        df_plot, df_plot_pred, df_plot_pred_all, str_date_last = \
            prepare_plot_data_pos(df_feat_fr, False)

        # prepare plot data for MAPS
        df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep, df_dep_sum = \
            prepare_plot_data_map(False)
    
        df_pos_fr = pd.read_csv(PATH_DF_POS_FR)
        df_pos_fr.index = df_pos_fr["date"]'''

        return get_title(value), "graph-mdl", "app-map", "app-graph-map", \
            "app-hide", "app-hide", "app-show-block", "app-hide"
            

# updata map / curve in KR : actions on buttons or dropdown
@app.callback([Output('loading-graph-map-kr', 'children'),
    Output('covid-rt-map-kr', 'figure'),
    Output('covid-rt-dep-graph-kr', 'figure'),
    Output('graph_type_kr', 'children'),
    Output('loading-output-kr', 'children')],
    [Input('btn-conf', 'n_clicks'),
    Input('country-dropdown', 'value'),
    Input('covid-rt-map-kr', 'clickData'),
    Input("btn-kr",'n_clicks'),
    Input("btn-test",'n_clicks'),
    Input("btn-rt",'n_clicks'), 
    Input('update-data', 'n_clicks')],
    [State("graph_type_kr", "children")],
    prevent_initial_call=True)
def update_map_kr_callback(n_clicks, dropdown_value, clickData, 
    n_clicks_country, n_clicks_test, n_clicks_rt, n_clicks_update, graph_type):


    if (dropdown_value != "South Korea"):
        raise PreventUpdate

    display_msg("update_map_kr_callback_callback...")

    # take context button
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print("button_id: ", button_id)

    str_date_kr = dash.no_update
    # if update button : update KR data only
    if (button_id == "update-data"):
        update_df_feat_kr()
    
    df_feat_kr = None
    
    # graph-type
    if (button_id == "btn-conf"):
        graph_type = 0
    elif (button_id == "btn-test"):
        graph_type = 1
    elif (button_id == "btn-rt"):
        graph_type = 2
    print("graph_type: ", graph_type)
    # check current dept.:
    try:
        if (button_id == "btn-kr"):
            dep_curr = "South Korea"
        else:
            dep_curr = clickData["points"][0]["location"]
            dep_curr = DICT_AREA[dep_curr]
    except:
        dep_curr = "South Korea"
    print(dep_curr)

    # figure curve confirmed
    if (graph_type == 0): 

        # figure map
        if (button_id != "covid-rt-map-kr"):
            if (df_feat_kr is None):
                df_feat_kr = load_df_feat_kr()
            fig_map = create_fig_map_kr(df_feat_kr, LIST_SUM_GEOJSON, 
                "<b>Confirmed</b> cases : Sum of last 14 days")
        else:
            fig_map = dash.no_update

        if (dep_curr == "South Korea"):
            if (df_feat_kr is None):
                df_feat_kr = load_df_feat_kr()
            ser_pos = df_feat_kr["pos"]
            ser_sum_pos = df_feat_kr["sum_cases"]
            rt_last = df_feat_kr["Rt"].values[-1]
            fig_curve = figure_pos(ser_pos, ser_sum_pos, dep_curr, rt_last)
        else:
            if (df_feat_kr is None):
                df_feat_kr = load_df_feat_kr()
            ser_pos = df_feat_kr[dep_curr]
            ser_sum_pos = df_feat_kr[f"sum_{dep_curr}"]
            rt_last = df_feat_kr[f"Rt_{dep_curr}"].values[-1]
            fig_curve = figure_pos(ser_pos, ser_sum_pos, dep_curr, rt_last)

    elif(graph_type == 1): # test pos rate
        # map
        fig_map = dash.no_update
        # curve
        if (df_feat_kr is None):
            df_feat_kr = load_df_feat_kr()
        fig_curve = create_fig_pos_rate(df_feat_kr, "South Korea")

    elif(graph_type == 2): # mode Rt

        # figure map
        if (button_id != "covid-rt-map-kr"):
            if (df_feat_kr is None):
                df_feat_kr = load_df_feat_kr()
            fig_map = create_fig_map_kr(df_feat_kr, LIST_RT_GEOJSON, 
            "<b>Rt</b> estimated for last 14 days")
        else:
            fig_map = dash.no_update

        if (dep_curr == "South Korea"):
            if (df_feat_kr is None):
                df_feat_kr = load_df_feat_kr()
            dep_curr = "South Korea"
            ser_rt = df_feat_kr[df_feat_kr["date"] >= "2020-03-19"]["Rt"]
            sum_pos_last = df_feat_kr["sum_cases"].values[-1]
            fig_curve = figure_rt(ser_rt, dep_curr, sum_pos_last, 
                country="South Korea")
            
        else:
            if (df_feat_kr is None):
                df_feat_kr = load_df_feat_kr()
            ser_rt = df_feat_kr[df_feat_kr["date"] >= \
                "2020-03-19"][f"Rt_{dep_curr}"]
            sum_pos_last = df_feat_kr[f"sum_{dep_curr}"].values[-1]
            fig_curve = figure_rt(ser_rt, dep_curr, sum_pos_last, 
                country="South Korea")

    if ((button_id == "update-data") & (df_feat_kr is not None)):
        str_date_kr = f'(up to: {df_feat_kr["date"].values[-1]})'
    
    return "", fig_map, fig_curve, graph_type, str_date_kr

# button update data
@app.callback(
    [Output('loading-output-1', 'children'), 
    Output('covid-pos-graph', 'figure'),
    Output('covid-rt-map', 'figure')],
    [Input('update-data', 'n_clicks')],
    [State('predicted-value', 'children'),
    State('predicted-value-all', 'children'),
    State('country-dropdown', 'value')], 
    prevent_initial_call=True)
def update_fr(n_clicks, jsonified_pred, jsonified_pred_all, dropdown_value):

    display_msg("UPDATE DATA BUTTON ...")
    
    if (dropdown_value != "France"):
        raise PreventUpdate

    if n_clicks < 1: # no update at loading
        display_msg("Nothing to do")
        display_msg("UPDATE DATA BUTTON END.")
        return dash.no_update

    flag_update = check_update()

    # prepare data input
    df_feat_fr, str_date_mdl, str_data_date = prepare_data_input(flag_update)
    
    # plot data for positive cases
    df_plot, df_plot_pred, df_plot_pred_all, str_date_last = \
        prepare_plot_data_pos(df_feat_fr, flag_update)

    # prepare plot data for MAPS
    df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep, _ = \
        prepare_plot_data_map(flag_update)

    display_msg("UPDATE DATA BUTTON END.")
    return str_data_date, \
        create_fig_pos(df_plot, df_plot_pred, df_plot_pred_all, str_date_mdl), \
        create_fig_map(pt_fr_test_last, dep_fr, str_date_last)




# click on map
@app.callback(
    [Output('loading-graph-map', 'children'),
    Output('covid-rt-dep-graph', 'figure'),
    Output('graph_type', 'children'), 
    Output('id_button', 'children'),
    Output('mode_country', 'children'),
    Output('dep', 'children')],
    [Input('covid-rt-map', 'clickData'), 
    Input('div-rt-map', 'n_clicks')], 
    [dash.dependencies.State('covid-rt-map', 'figure'),
    dash.dependencies.State('graph_type', 'children'),
    dash.dependencies.State('id_button', 'children'),
    dash.dependencies.State('mode_country', 'children'),
    dash.dependencies.State('dep', 'children'),
    dash.dependencies.State('covid-rt-map', 'hoverData')])
def display_click_data(clickData, n_clicks, fig, graph_type_old, id_button_old,
    mode_country_old, dep_old, hoverData):
    display_msg(" ")
    display_msg("display_click_data ...")
    print("clickData: ", clickData)
    print("hoverData: ", hoverData)
    
    # get context : 
    # if "div-rt-map" => user clicked on buttons 
    # if "covid-rt-map" => user clicked on data
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print("button_id: ", button_id)

    # check current dept.:
    try:
        dep_curr = clickData["points"][0]["location"]
    except:
        dep_curr = "Paris"
    print(dep_curr)

    # treat graph type :
    (a,) = fig["layout"]["updatemenus"]
    id_button = a["active"]
    if id_button < 3:
        graph_type = id_button
    else:
        graph_type = graph_type_old

    print('id_button (old): ', id_button_old)
    print('id_button (new): ', id_button)
    print("graph_type (old):", graph_type_old)
    print("graph_type (new):", graph_type)
    
    # if click change to France button, display Country not dept.
    if (id_button_old !=4) & (id_button==4):
        mode_country = 1
        if (dep_curr != dep_old):
            mode_country = 0
        else:
            mode_country = 1
    # if no change for buttons or graph type but change dept
    # then displays dept.
    elif (id_button_old == id_button) & (graph_type_old == graph_type):
        if (dep_curr != dep_old):
            mode_country = 0
        else:
            mode_country = 1
    elif (id_button_old != id_button) & (id_button == 0):
        mode_country = 0
    else:
        mode_country = mode_country_old
    print("mode_country: ", mode_country)

    # buttons Zoom (IDF or DOM-TOM) (prevent update)
    if ((id_button == 3) | (id_button == 5)) & (button_id == "div-rt-map"):
        print("PreventUpdate.")
        return dash.no_update, dash.no_update, graph_type, id_button, \
            mode_country, dep_curr


    # si type graph Rt
    if (graph_type == 2):
        if (mode_country == 1):
            # load from disk
            df_feat_fr = load_data_pos()
            fig_out = create_fig_rt_fr(df_feat_fr)
        else:
            # prepare plot data for MAPS
            df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep, _ = \
                prepare_plot_data_map(False)
            fig_out = create_fig_rt_dep(dep_curr, df_code_dep, 
                    pt_fr_test_last, df_dep_r0)
    elif (graph_type == 1):
        if (graph_type_old != 1):
            df_feat_fr = load_data_pos()
            fig_out = create_fig_pos_rate(df_feat_fr)
        else:
            print("PreventUpdate.")
            return dash.no_update, dash.no_update, graph_type, id_button, \
                mode_country, dep_curr
    # user in mode Confirmed, only dept. available 
    else:
        # if user in mode "Confirmed"
        # 
        # prepare plot data for MAPS
        if not mode_country:
            df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep, df_dep_sum = \
                prepare_plot_data_map(False)
            df_pos_fr = pd.read_csv(PATH_DF_POS_FR)
            df_pos_fr.index = df_pos_fr["date"]
            fig_out = create_fig_pos_dep(dep_curr, df_code_dep, 
                        df_dep_r0, df_pos_fr, df_dep_sum)
        else:
            print("PreventUpdate.")
            return dash.no_update, dash.no_update, graph_type, id_button, \
                mode_country, dep_curr

    if dep_curr is None:
        dep_curr =""

    display_msg("display_click_data END.")
    return "", fig_out, graph_type, id_button, mode_country, dep_curr
    

if __name__ == '__main__':
    #app.run_server(debug=True)
    app.config.suppress_callback_exceptions = True

    # if mode test preparation app

    if os.getenv("APP_MODE_ENV") is not None:
        if os.getenv("APP_MODE_ENV") == "TEST":
            print("Test App Preparation OK.")
            sys.exit()

    print("Run server :")
    app.run_server(host='0.0.0.0', debug=settings.MODE_DEBUG, port=80)

    #raise RuntimeError('Not running with the Werkzeug Server')


