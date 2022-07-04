# -*- coding: utf-8 -*-
'''
Helpers for plots
'''
# import bluit-in
import math
# import third party
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import project modules 
from my_helpers.utils import display_msg
from my_helpers.model import FUTURE_TARGET
from my_helpers.model import PAST_HISTORY
from my_helpers.data_plots_kr import GEOJSON_KR
from my_helpers.data_plots_kr import LIST_NAME_GEOJSON
from my_helpers.data_plots_kr import LAT_LON_KR
from my_helpers.data_plots_kr import ZOOM_KR

def create_fig_pos(df_plot, df_plot_pred, df_plot_pred_all, str_date_mdl, 
    area="France"):
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
    title_fig = f'<b>COVID-19 Confirmed cases in {area}</b>' + \
        '<br>Model trained until <b>' + str_date_mdl + '</b>' + \
        '<br>predicts next {} days with last {} days until <b>' \
        .format(FUTURE_TARGET, PAST_HISTORY) + \
        df_plot_pred["date"].max() + '</b>'
    fig.update_layout(title=title_fig, yaxis_title='nb <b>Total</b> cases')
    fig.update_layout(legend_orientation="h", legend=dict(x=0, y=1))
    #fig.update_layout(height=600)

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
        #height=600,
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
    #fig.update_layout(height=600)
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
    #fig.update_layout(height=600)
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
    #fig.update_layout(height=600)
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
    #fig.update_layout(height=600)
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
    #fig.update_layout(height=600)
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

