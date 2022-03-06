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
#import logging
import datetime
import os
import sys

# import third party 

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input
from dash.dependencies import Output
from dash.dependencies import State
from dash.exceptions import PreventUpdate
import pandas as pd
#from aws_logging_handlers.S3 import S3Handler
# import project modules 
import settings
from my_helpers.dates import *
from my_helpers.data_plots import prepare_data_input
from my_helpers.data_plots import prepare_plot_data_pos
from my_helpers.data_plots import check_update
from my_helpers.data_plots import PATH_DF_POS_FR
from my_helpers.data_plots import load_data_pos
from my_helpers.data_plots_kr import load_df_feat_kr
from my_helpers.data_plots_kr import LIST_SUM_GEOJSON
from my_helpers.data_plots_kr import LIST_RT_GEOJSON
from my_helpers.data_plots_kr import DICT_AREA
from my_helpers.data_plots_kr import check_update_kr
from my_helpers.data_plots_kr import update_df_feat_kr
from my_helpers.data_plots_kr import prepare_data_input_kr
from my_helpers.data_plots_kr import prepare_plot_data_pos_kr
from my_helpers.plots import figure_pos
from my_helpers.plots import create_fig_pos
from my_helpers.plots import figure_rt
from my_helpers.plots import create_fig_rt
from my_helpers.plots import create_fig_map
from my_helpers.plots import create_fig_map_kr
from my_helpers.plots import create_fig_rt_dep
from my_helpers.plots import create_fig_rt_fr
from my_helpers.plots import create_fig_pos_dep
from my_helpers.plots import create_fig_pos_rate
#from my_helpers.model import calc_rt
#from my_helpers.model import NB_DAYS_CV
#from my_helpers.model import FUTURE_TARGET, PAST_HISTORY
from my_helpers.data_maps import prepare_plot_data_map
#from my_helpers.utils import sum_mobile
from my_helpers.utils import display_msg
# DEFINITIONS

PATH_TO_SAVE_DATA = settings.PATH_TO_SAVE_DATA
BUCKET_LOGS = settings.BUCKET_NAME

meta_tags=[{
      'name': 'viewport',
      'content': 'width=device-width, initial-scale=1.0'
    }
]

# informations
markdown_info_mdl = '''
***Legend***    
`Total`                 : Actual total number of confirmed cases in France for past days  
`Total (estim.)`        : Estimated total number of confirmed cases in France for past days (by model)  
`Total (future estim.)` : Estimated total number of confirmed cases in France for future days (by model)  
`Daily`                 : Actual daily number of confirmed cases in France for past days  
`Daily (future estim.)` : Estimated daily number of confirmed cases in France for future days (by model)  
`Daily (estim.)`        : Estimated daily number of confirmed cases in France for past days (by model)  

***About Model***  
    
The prediction model is a new TCN Deep Learning Tensorflow model.  
    
TCN : Temporal Convolutional Network (https://github.com/philipperemy/keras-tcn)
    
It estimates the number of daily confirmed cases in France for next days by time-series forecast.  
    
API France data from French Gouvernment are 4 days in late...  
  
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

If predictions are under-estimated, the actual evolution is going in a "BAD" way...  
If predictions are over-estimated, the actual evolution is going in a "GOOD" way...  
    
But more the model learns from the different "waves", more the model is accurate.  

***DATA sources***: 
- Tested / Confirmed cases FR: https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-resultats-des-tests-virologiques-covid-19
- Météo France: https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm
    
'''  
markdown_info_mdl_kr = '''
***Legend***    
`Total`                 : Actual total number of confirmed cases in France for past days  
`Total (estim.)`        : Estimated total number of confirmed cases in France for past days (by model)  
`Total (future estim.)` : Estimated total number of confirmed cases in France for future days (by model)  
`Daily`                 : Actual daily number of confirmed cases in France for past days  
`Daily (future estim.)` : Estimated daily number of confirmed cases in France for future days (by model)  
`Daily (estim.)`        : Estimated daily number of confirmed cases in France for past days (by model)  

***About Model***  
    
The model is a simple LSTM Deep Learning Tensorflow model.  
    
It estimates the number of daily confirmed cases in South Korea for next days by time-series forecast.  
    
For that, the model takes a period of 14 days to estimate the next 7 days.  
    
Because of lack of data, it has been trained with only few past periods and validated on only very few periods!  
    
Input Features are daily data for:
- Min/Max Temperatures
- Mean Humidities
- Wind speed
- Confirmed cases
- Tested cases
- Day of the week
- Mean Age of Confirmed cases

If predictions are under-estimated, the actual evolution is going in a "BAD" way...  
If predictions are over-estimated, the actual evolution is going in a "GOOD" way...  
    
But more the model learns from the different "waves", more the model is accurate.  

***DATA sources***: 
- Tested / Confirmed cases KR: https://www.data.go.kr/tcs/dss/selectApiDataDetailView.do?publicDataPk=15043376
- Confirmed cases by age KR: https://www.data.go.kr/tcs/dss/selectApiDataDetailView.do?publicDataPk=15043377
- Confirmed cases by area KR: https://www.data.go.kr/tcs/dss/selectApiDataDetailView.do?publicDataPk=15043378
- Meteo South Korea (Seoul, Deagu, Busan): https://www.visualcrossing.com/weather-data
- Geoson Map South Korea: https://github.com/southkorea/southkorea-maps 
  
'''  

markdown_info = '''  
More info in my github below.  
    
***GitHub***: https://github.com/jeugregg/coronavirusModel
    
'''
# HELPER FUNCTIONS

def get_title(str_country):
    return f'COVID-19 in {str_country} Dashboard: ' + \
            'Datavisualization & Model'

def jsonifed_pred(df_plot_pred):
    '''Transform DataFrame into json with date '''
    return df_plot_pred.to_json(date_format='iso', orient='split')

#
# APP DASH
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, 
    meta_tags=meta_tags)
app.title = "App Covid Visu"
server = app.server

#logger = logging.getLogger('root')
#logger.setLevel(logging.INFO)
#s3_handler = S3Handler("logs/test_log", BUCKET_LOGS, workers=1, 
#    encryption_options={}, max_file_size_bytes=1024)
#formatter = logging.Formatter('[%(asctime)s] %(filename)s:%(lineno)d}' + \
#    ' %(levelname)s - %(message)s')
#s3_handler.setFormatter(formatter)

#logger.addHandler(s3_handler)
#logger.info("test info message")
#logger.warning("test warning message")
#logger.error("test error message")
'''
                html.Button(
                    'Force Update!', 
                    id='force-update-button', 
                    n_clicks=0,
                ),
'''
def create_error_msg(e):
    return html.Div([
                dcc.Textarea(
                    id='textarea-logs',
                    value=f"{e}",
                    style={'width': '100%', 'height': 200},
                ),

            ])

def startup_layout(force_update=None, message=""):
    '''
    startup web page
    '''
    display_msg("STARTUP...")
    #logger.info("STARTUP...")
    try:
        # update
        if settings.MODE_FORCE_UPDATE:
            flag_update = True
            flag_update_kr = True
        elif force_update is not None:
            flag_update = force_update
            flag_update_kr = force_update
        else:
            flag_update = check_update()
            flag_update_kr = check_update_kr()
        
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
        
        # prepare input data KR
        df_feat_kr, str_date_mdl_kr, str_data_date_kr = \
            prepare_data_input_kr(flag_update=flag_update_kr)
        
        # prepare plot data for positive cases KR
        df_plot_kr, df_plot_pred_kr, df_plot_pred_all_kr, str_date_last_kr = \
            prepare_plot_data_pos_kr(df_feat_kr, flag_update=flag_update_kr)

        str_country = "France"

        display_msg("STARTUP END.")
        #logger.info("STARTUP END.")
    except Exception as e:
        #logger.error("ERROR STARTUP: " + str(e))

        raise(e)
    

    return html.Div(children=[
        dcc.Location(id='url', refresh=False),
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
                html.Div(id="loading-output-kr", children=str_data_date_kr, 
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
                    dcc.Graph(id='covid-pos-graph-kr', className="app-hide",
                        figure=create_fig_pos(df_plot_kr, df_plot_pred_kr,
                            df_plot_pred_all_kr, str_date_mdl_kr,
                            area="South-Korea")),
                    html.Div(id='info_mdl',
                        children=dcc.Markdown(children=markdown_info_mdl))
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
        html.Div(id='info', 
            children=dcc.Markdown(children=markdown_info)),
        create_error_msg(message),
        ])

try:
    app.layout = startup_layout(
            force_update=None,
            message="App : OK",
        )
except Exception as e1:
    try:
        app.layout = startup_layout(
            force_update=True,
            message=e1,
        )
    except Exception as e2:

        app.layout = create_error_msg(str(e1) + "\n" + str(e2))

@app.callback([Output("country-dropdown", "value")],
[Input('url', 'pathname')])
def location_url_callback(pathname):
    print('url pathname: ', pathname)
    if (pathname == "/South-Korea"):
        value = "South Korea"
    elif (pathname == "/")|(pathname == "/France"):
        value = "France"
    else:
        value = dash.no_update
    return [value]

@app.callback(
    [Output("app-title", "children"),
    Output('covid-pos-graph', 'className'),
    Output('covid-pos-graph-kr', 'className'),
    dash.dependencies.Output("div-rt-map", "className"),
    dash.dependencies.Output("div-rt-curve", "className"),
    dash.dependencies.Output("div-rt-map-kr", "className"),
    dash.dependencies.Output("div-rt-curve-kr", "className"),
    Output("loading-output-1", "className"),
    Output("loading-output-kr", "className"),
    Output("info_mdl", "children")],
    [dash.dependencies.Input('country-dropdown', 'value')],
    prevent_initial_call=True)
def update_tabs(value):
    
    display_msg("update_tabs ...")
    #logger.info("update_tabs ...")
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

        
        return get_title(value), "app-hide", "graph-mdl", "app-hide", \
            "app-hide", "app-map", "app-graph-map", "app-hide", \
            "app-show-block", dcc.Markdown(children=markdown_info_mdl_kr)
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

        return get_title(value), "graph-mdl", "app-hide", "app-map", \
            "app-graph-map", "app-hide", "app-hide", "app-show-block", \
            "app-hide", dcc.Markdown(children=markdown_info_mdl)
            

# update map / curve in KR : actions on buttons or dropdown
@app.callback([Output('loading-graph-map-kr', 'children'),
    Output('covid-rt-map-kr', 'figure'),
    Output('covid-rt-dep-graph-kr', 'figure'),
    Output('graph_type_kr', 'children'),
    Output('loading-output-kr', 'children'),
    Output('covid-pos-graph-kr', 'figure')],
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
    #logger.info("update_map_kr_callback_callback...")
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
        #update_df_feat_kr()
        # prepare input data KR (update if available)
        df_feat_kr, str_date_mdl_kr, str_data_date_kr = \
            prepare_data_input_kr(flag_update=True)
    else:
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
        
        # prepare plot data for positive cases KR
        df_plot_kr, df_plot_pred_kr, df_plot_pred_all_kr, str_date_last_kr = \
            prepare_plot_data_pos_kr(df_feat_kr, flag_update=True)
        # update graph pos mdl
        fig_mdl = create_fig_pos(df_plot_kr, df_plot_pred_kr, 
            df_plot_pred_all_kr, str_date_mdl_kr, "South-Korea")

        return "", fig_map, fig_curve, graph_type, str_date_kr, fig_mdl
    else:
        # no need to update graph pos mdl
        return "", fig_map, fig_curve, graph_type, str_date_kr, dash.no_update

    
    

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
    #logger.info("UPDATE DATA BUTTON ...")
    if (dropdown_value != "France"):
        raise PreventUpdate

    if n_clicks < 1: # no update at loading
        display_msg("Nothing to do")
        #logger.info("Nothing to do")
        display_msg("UPDATE DATA BUTTON END.")
        #logger.info("UPDATE DATA BUTTON END.")
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
    #logger.info("UPDATE DATA BUTTON END.")
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
    #logger.info("display_click_data ...")
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
    #logger.info("display_click_data END.")
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


