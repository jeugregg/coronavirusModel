# coronavirusModel
 Coronavirus Visualization & Modeling

This project is about COVID-19 evolution.

I have developed 3 main python projects.

1) A very complete notebook : [coronavirus-visualization-modeling.ipynb](https://github.com/jeugregg/coronavirusModel/blob/master/coronavirus-visualization-modeling.ipynb)  
This notebook shows the evolution of COVID-19 virus all over the world.  

It also focuses on South Korea and France areas.  

Animated Maps are Available at Region-level for South Korea, France, USA, China.  
Also, more globally, at country-level for all other countries in the World.  
 
Also, this notebook scraps data from French and Korean Health official websites.  
If you discover the code, you can see how.  
Korean & French data are updated daily.  

Published version is available on kaggle : https://www.kaggle.com/jeugregg/coronavirus-visualization-modeling

The world data source is https://github.com/CSSEGISandData/COVID-19 provided by JHU CSSE

South Korea areas data are retrieved with scrapy from KCDC Press Release articles at https://www.cdc.go.kr/board/board.es?mid=a30402000000&bid=0030.

2) App Dashboard with evolution prediction by Deep Learning: [app.py](https://github.com/jeugregg/coronavirusModel/blob/master/app.py)

This app is online here : http://app-covid-visu.coolplace.fr/

I added a simple LSTM Deep Learning Tensorflow model to estimate actual total confirmed cases number in France for a very close-time time-series forecast.

It is developed in [Plotly Dash](https://plotly.com/dash/) 

You can see the model development notebook [ModelCovidTimeSeries.ipynb](https://github.com/jeugregg/coronavirusModel/blob/master/ModelCovidTimeSeries.ipynb)

The model use last 10 days period in the past to estimate the next 3 days in the future.  
Because of lack of data, it has been trained with only 70 past periods and validated on only 4 periods!  

Input Features are daily data for last 10 days:  
- Min/Max Temperatures
- Min/Max Humidities
- Comfirmed cases
- Test cases
- Day of the week

The predictions are under-estimated because the evolution are big during last days.  
The model will learn from this current changing period in few weeks, and it will be better.  
If new data is available, and at each call, model predict confirmed cases for 3 next days.  

The model is hosted on AWS EC2 Cluster (t2.micro).  
Because memory needed to predict is too high on this t2.micro instance, I use AWS Lambda API call.  
I use [serverless framework](https://www.serverless.com/).  
The tensorflow model is converted in Tensorflow LITE to respect storage limit for AWS lambda function.  
For conversion, it not so easy because of LSTM neural network need special format.  
Have a look at this [tutorial](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/experimental_new_converter/Keras_LSTM_fusion_Codelab.ipynb).

And Tensorflow lite library is pre-compiled before packaging it in lambda function.  

If you are interested, look at this [very good tutorial](https://github.com/edeltech/tensorflow-lite-on-aws-lambda).


DATA Sources : 
- Tested / Confirmed cases: https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-resultats-des-tests-virologiques-covid-19
- Meteo France : https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm


3) Deep Learning for read Table in HTML : [readTableWithBERT.ipynb](https://github.com/jeugregg/coronavirusModel/blob/master/readTableWithBERT.ipynb)
Additional file : [read Table into HTML BERT model resume training.ipynb](https://github.com/jeugregg/coronavirusModel/blob/master/read%20Table%20into%20HTML%20BERT%20model%20resume%20training.ipynb)  
I stop this project because I just tried transfert learning from a BERT style model : distilbert and I had bad results.

I prefer adapting scrapy classical method every times table format changed (KCDC table COVID-19 reports).

I used this github to train the model : [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers)

I think it is not the good model to do that.










