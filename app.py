# -*- coding: utf-8 -*-

''' App Covid Visu web app
COVID-19 in France Dashboard: Datavisualization & Model
Map of Reproduction number in France by "départements"
A tensorflow deep leanring model try to estimate next 7 days confirmed cases in 
France with last 14 days data available.
'''
# Run this app with `python app.py` and
# visit http://0.0.0.0/ in your web browser.

import settings
# General Definition 
#MODE_DEBUG = True # default = False 
#MODE_FORCE_UPDATE = False # default = False 
#PREDICT = True # default = True 
#MODEL_TFLITE = True # default = True 
PATH_TO_SAVE_DATA = settings.PATH_TO_SAVE_DATA

import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import datetime

import re
import os

# Project modules 

from my_helpers.dates import *

from my_helpers.data_plots import prepare_data_input
from my_helpers.data_plots import prepare_plot_data_pos
from my_helpers.data_plots import check_update

from my_helpers.model import FUTURE_TARGET, PAST_HISTORY

from my_helpers.data_maps import prepare_plot_data_map

# DEFINITIONS


#NB_POS_DATE_MIN_DF_FEAT = 140227 # on 12/05/2020
#date_format = "%Y-%m-%d"


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
    # new cases
    fig.add_trace(go.Bar(x=df_plot["date"].astype(np.datetime64), 
                        y=df_plot["pos"], 
                        name="Daily", opacity=0.5), 
                secondary_y=True)
    # total
    fig.add_trace(go.Scatter(x=df_plot["date"].astype(np.datetime64), 
                            y=df_plot["nb_cases"],
                        mode='lines+markers',
                        line_shape='linear',
                        connectgaps=True, name="Total"),
                secondary_y=False)

    fig.add_trace(go.Scatter(x=df_plot_pred_all["date"].astype(np.datetime64), 
                            y=df_plot_pred_all["nb_cases"],
                        mode='lines+markers',
                        line_shape='hv',
                        connectgaps=True, name="Total (estim.)"),
                secondary_y=False)

    fig.add_trace(go.Scatter(x=df_plot_pred["date"].astype(np.datetime64), 
                            y=df_plot_pred["nb_cases"],
                        mode='lines+markers',
                        line_shape='hv',
                        connectgaps=True, name="Total (future estim.)"),
                secondary_y=False)


    fig.add_trace(go.Bar(x=df_plot_pred["date"].astype(np.datetime64), 
                y=df_plot_pred["pos"], 
                name="Daily (future estim.)", opacity=0.5), 
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

    fig.update_yaxes(title_text="nb <b>Daily</b> cases", 
                    range=[0, 25000], secondary_y=True)
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
    lat_lon_fr =  {'lat':  46, 'lon': 2}
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
    display_msg("create_fig_map END.")
    return fig

def create_fig_rt_dep(dep_curr, df_code_dep,pt_fr_test_last, df_dep_r0):
    
    '''Rt evolution plots for one departement
    
     data : 
     - df_dep_r0 (date,  date / dep. Rt)
     - df_code_dep (-, dep. code / dep name )
     - pt_fr_test_last (-, p / t / dep / code / name / p_0 / R0)
    '''
    display_msg("create_fig_rt_dep ...")
    dep_num_curr = df_code_dep.loc[df_code_dep["name"] == dep_curr, 
                            "code"].values[0]

    if (df_dep_r0[dep_num_curr][-1] > 1) & \
        (pt_fr_test_last.loc[ \
            pt_fr_test_last.dep == dep_num_curr, "p"].values[0] > 400):
        color_curr = "red"
    elif (df_dep_r0[dep_num_curr][-1] > 1):
        color_curr = "orange"
    else:
        color_curr = "blue"
        
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_dep_r0["date"], y=df_dep_r0[dep_num_curr],
                mode='lines', name=dep_curr, line=dict(color=color_curr),
                fill='tozeroy'))

    fig.add_trace(go.Scatter(x=[df_dep_r0["date"][0], 
                                df_dep_r0["date"][-1]], 
                            y=[1,1],
                            mode='lines', 
                            line=dict(color="red", dash='dash'),
                            hoverinfo="skip"))


    fig.add_annotation(
                x=0,
                y=-0.18,
                text="<i>(Click on Map to Update this Curve)</i>")
    fig.update_annotations(dict(
                xref="paper",
                yref="paper",
                showarrow=False
    ))

    subtitle_curr = "Reproduction nb. Rt: " + \
                    "<b>{:.2f}</b> ".format(df_dep_r0[dep_num_curr][-1]) + \
                    '({})<br>'.format(df_dep_r0['date'].max())  + \
                    "Confirmed cases: <b>{}</b>".format(pt_fr_test_last.loc[ \
                    pt_fr_test_last.dep == dep_num_curr, "p"].values[0]) + \
                        " for 14 days"

    fig.update_layout(title="<b>{}</b> : ".format(dep_curr) + '<br>' + \
        subtitle_curr,
        showlegend=False,
        font=dict(
            size=12,
        )
    )
    display_msg("create_fig_rt_dep END.")
    return fig

# APP DASH
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "App Covid Visu"



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
    df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep = \
        prepare_plot_data_map(flag_update)
    
    # informations
    markdown_info = '''
    ***Legend***  
    `Daily`                 : Actual daily number of confirmed cases in France for past days  
    `Total`                 : Actual total number of confirmed cases in France for past days  
    `Total (estim.)`        : Estimated total number of confirmed cases in France for past days (by model)  
    `Total (future estim.)` : Estimated total number of confirmed cases in France for future days (by model)  
    `Daily (future estim.)` : Estimated daily number of confirmed cases in France for future days (by model)  
    
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
    display_msg("STARTUP END.")
    return html.Div(children=[
        html.H1(children='COVID-19 in France Dashboard: ' + \
            'Datavisualization & Model'),
        html.Div(children=html.Button('Update Data', id='update-data', 
        n_clicks=0), style={'display': 'inline-block', 'margin-right': 10}),
        html.Div(children=dcc.Loading(
            id="loading-fig-pos",
            type="default",
            children=html.Div(id="loading-output-1", children=str_data_date)), 
            style={'display': 'inline-block', 'margin-right': 10}),
            
        html.Div(children=html.A(
            children="By Gregory LANG, Data Scientist Freelance",
            href="http://greg.coolplace.fr/data-scientist-freelance", 
            target="_blank"), style={'display': 'inline-block'}),

        dcc.Tabs(id='tabs-example', value='tab-1', children=[
            dcc.Tab(label='Evolution & Model', value='tab-1', children=[
            dcc.Graph(id='covid-pos-graph',
            figure=create_fig_pos(df_plot, df_plot_pred, df_plot_pred_all, 
                str_date_mdl), style={'margin-top': 10})
            ]),
            dcc.Tab(label='Maps', 
                value='tab-2', children=[
                html.Div(id="div-rt-map", children=dcc.Graph(id='covid-rt-map',
            figure=create_fig_map(pt_fr_test_last, dep_fr, str_date_last), 
                ), style={'width': '59%', 'display': 'inline-block', 
                    'margin-right': 1}, n_clicks=0),
                html.Div(dcc.Graph(id='covid-rt-dep-graph',
            figure=create_fig_rt_dep("Paris", df_code_dep,pt_fr_test_last, 
                df_dep_r0)), style={'width': '39%', 'display': 'inline-block', 
                'margin-top': 10})
            ])
        ], style={'margin-top': 10}),
        # Hidden div inside the app that stores the intermediate value
        html.Div(id='predicted-value', style={'display': 'none'},
            children=jsonifed_pred(df_plot_pred)),
        html.Div(id='predicted-value-all', style={'display': 'none'},
            children=jsonifed_pred(df_plot_pred_all)),
        html.Div(id='info', children=dcc.Markdown(children=markdown_info))
        ])

app.layout = startup_layout


# button update data
@app.callback(
    [dash.dependencies.Output('loading-output-1', 'children'), 
    dash.dependencies.Output('covid-pos-graph', 'figure'),
    dash.dependencies.Output('covid-rt-map', 'figure')],
    [dash.dependencies.Input('update-data', 'n_clicks')],
    [dash.dependencies.State('predicted-value', 'children'),
    dash.dependencies.State('predicted-value-all', 'children')])
def load_figure(n_clicks, jsonified_pred, jsonified_pred_all):
    display_msg("UPDATE DATA BUTTON ...")
    
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
    df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep = \
        prepare_plot_data_map(flag_update)

    display_msg("UPDATE DATA BUTTON END.")
    return str_data_date, \
        create_fig_pos(df_plot, df_plot_pred, df_plot_pred_all, str_date_mdl), \
        create_fig_map(pt_fr_test_last, dep_fr, str_date_last)

# click on map
@app.callback(
    Output('covid-rt-dep-graph', 'figure'),
    [Input('covid-rt-map', 'clickData')], 
    [dash.dependencies.State('div-rt-map', 'n_clicks')])
def display_click_data(clickData, n_clicks):
    display_msg("display_click_data ...")
    # don't work  : block first click on map (need to click on buttons before)
    '''if n_clicks < 1:
        display_msg("Nothing to do!")
        display_msg("display_click_data END.")
        return dash.no_update'''

    try:
        dep_curr = clickData["points"][0]["location"]
    except:
        dep_curr = "Paris"

    # prepare plot data for MAPS
    df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep = \
        prepare_plot_data_map(False)
    display_msg("display_click_data END.")
    return create_fig_rt_dep(dep_curr, df_code_dep, 
                pt_fr_test_last, df_dep_r0)

if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server(host='0.0.0.0', debug=settings.MODE_DEBUG, port=80)
    app.config.suppress_callback_exceptions = True

