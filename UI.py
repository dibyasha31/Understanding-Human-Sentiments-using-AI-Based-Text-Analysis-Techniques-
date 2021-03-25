# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 20:40:48 2021

@author: Divyasha Pradhan
"""

# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
import numpy as np
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import webbrowser
from io import BytesIO
from wordcloud import WordCloud
import base64



app = dash.Dash(external_stylesheets=[dbc.themes.SUPERHERO])
project_name = "Sentiment Analysis with Insights"

    
def load_model():
    global scrappedReviews
    scrappedReviews = pd.read_csv('scrappedReviews.csv')
    
    global balanced_reviews
    balanced_reviews=pd.read_csv('balanced_reviews.csv')
    
    global pickle_model
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)

    global vocab
    file = open("feature.pkl", 'rb') 
    vocab = pickle.load(file)

def check_review(reviewText):

    #reviewText has to be vectorised, that vectorizer is not saved yet
    #load the vectorize and call transform and then pass that to model preidctor
    #load it later

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))


    # Add code to test the sentiment of using both the model
    # 0 == negative   1 == positive
    
    return pickle_model.predict(vectorised_review)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')
    
def create_app_ui():
    global balanced_reviews
   
    balanced_reviews = balanced_reviews.dropna()
        # df = df[df['overall'] != 3]
    balanced_reviews['Positivity'] = np.where(
            balanced_reviews['overall'] > 3, 1, 0)
    labels = ['Positive Reviews', 'Negative Reviews', 'Neutral Reviews']
    values = [balanced_reviews[balanced_reviews.overall > 3].dropna().shape[0],balanced_reviews[balanced_reviews.overall < 3].dropna(
        ).shape[0],balanced_reviews[balanced_reviews.overall == 3].dropna().shape[0]]
    labels1 = ['+ve Reviews', '-ve Reviews']
    values1 = [len(balanced_reviews[balanced_reviews.Positivity == 1]), len(
            balanced_reviews[balanced_reviews.Positivity == 0])]


    colors = ['#660033', '#ffff00', '#990000']
    
    main_layout = dbc.Container(
        dbc.Jumbotron(
                [   
                    html.H1(id='heading1', children='Sentiment Analysis Of Etsy.com',
                            className='display-3 mb-4', style={'font': 'sans-seriff', 'font-weight': 'bold', 'font-size': '50px', 'color': 'black'}),
                    
                    html.P(id='heading5', children='Distribution of reviews based on filtered data',
                           className='display-3 mb-4', style={'font': 'sans-seriff', 'font-weight': 'bold', 'font-size': '30px', 'color': 'black'}),
                    
                    dbc.Container(
                        dcc.Loading(
                            dcc.Graph(
                                figure={'data': [go.Pie(labels=labels, values=values, hole=.3, pull=[0.2, 0, 0], textinfo='value', marker=dict(colors=colors, line=dict(color='#000000', width=2)))],
                                        'layout': go.Layout(height=600, width=1000, autosize=False)
                                        }
                            )
                        ),
                        className='d-flex justify-content-center'
                    ),

                    html.Hr(),
                    html.P(id='heading5', children='The Positivity Measure',
                           className='display-3 mb-4', style={'font': 'sans-seriff', 'font-weight': 'bold', 'font-size': '30px', 'color': 'black'}),
                    
                    dbc.Container(
                        dcc.Loading(
                            dcc.Graph(
                                id='example-graph',
                                figure={
                                    'data': [
                                        go.Bar(y=labels1, x=values1, orientation='h', marker=dict(
                                            color="turquoise"))
                                    ],
                                    'layout': go.Layout(xaxis={'title': 'Sentiments'}, yaxis={'title': 'Emotions'}),
                                }
                            )
                        ),
                    ),

                    html.Hr(),
                    
                            html.Div(
                    [   
                        html.Div([
                        html.H2(id='heading4',children='Word Cloud', className='display-3 mb-4', style={'font': 'sans-seriff', 'font-weight': 'bold', 'font-size': '30px', 'color': 'black'}),
                        dbc.Button("ALL Words",
                         id="allbt",
                         outline=True,
                         color="info", 
                         className="mr-1",
                         n_clicks_timestamp=0,
                         style={'padding':'10px','padding-right':'15px'}
                         ),
                        dbc.Button("Positve Words",
                        id="posbt",
                         outline=True,
                         color="success", 
                         className="mr-1",
                         n_clicks_timestamp=0,
                         style={'padding':'10px','padding-right':'15px'}
                         ),
                        dbc.Button("Negative Words",
                        id="negbt",
                        outline=True, 
                        color="danger",
                        className="mr-1",
                        n_clicks_timestamp=0,
                        style={'padding':'10px','padding-right':'15px'}
                        )
                        ],style={'padding-left':'15px'}
                        ),
                        html.Div(id='container',style={'padding':'15px'})
                    ]
                ),
                    
                    html.H1(id = 'heading', children = project_name, className = 'display-3 mb-4',style={'font': 'sans-seriff', 'font-weight': 'bold', 'font-size': '30px', 'color': 'black'}),
                    dbc.Textarea(id = 'textarea', className="mb-3", placeholder="Enter the Review", value = 'I hate these shoes', style = {'height': '150px'}),
                    
                    dbc.Container([
                        dcc.Dropdown(
                    id='dropdown',
                    placeholder = 'Select a Review',
                    options=[{'label': i[:100] + "...", 'value': i} for i in scrappedReviews.reviews],
                    value = scrappedReviews.reviews[0],
                    style = {'margin-bottom': '30px'}
                    
                    
                )   
                       ],
                        style = {'padding-left': '50px', 'padding-right': '50px'}
                        ),
                    dbc.Button("Submit", color="dark", className="mt-2 mb-3", id = 'button', style = {'width': '100px'}),
                    html.Div(id = 'result'),
                    html.Div(id = 'result1')
                    ],
                className = 'text-center'
                ),
               className='mt-4'
        )
    
    return main_layout

@app.callback(
    Output('result', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    
def update_app_ui(n_clicks, textarea):
    result_list = check_review(textarea)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")

@app.callback(
    Output('result1', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
     State('dropdown', 'value')
     ]
    )
def update_dropdown(n_clicks, value):
    result_list = check_review(value)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")
    
@app.callback(
    Output('container','children'),
    [
        Input('allbt','n_clicks_timestamp'),
        Input('posbt','n_clicks_timestamp'),
        Input('negbt','n_clicks_timestamp'),
    ]
)
def wordcloudbutton(allbt,posbt,negbt):

    if int(allbt) > int(posbt) and int(allbt)>int(negbt):
        return html.Div([
            html.Img(src=app.get_asset_url('all.png'))])
    elif int(posbt) > int(allbt) and int(posbt)>int(negbt):
        return html.Div([
            html.Img(src=app.get_asset_url('positive.png'))
            ])
    elif int(negbt) > int(allbt) and int(negbt) > int(posbt):
       return html.Div([
           html.Img(src=app.get_asset_url('negative.png'))
           ])
    else:
        pass

def main():
    global app
    global project_name
    load_model()
    open_browser()
    app.layout = create_app_ui()
    app.title = project_name
    app.run_server(debug=True)
    app = None
    project_name = None
if __name__ == '__main__':
    main()