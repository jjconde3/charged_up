# Databricks notebook source
# MAGIC %md
# MAGIC # Import Libraries

# COMMAND ----------

pip install markupsafe==2.0.1

# COMMAND ----------

# import libraries
import pandas as pd
# import altair as alt
import datetime as dt
import numpy as np
#import json
import time
#import urllib
#import requests
import seaborn as sns
#from vega_datasets import data
#from pandas.io.json import json_normalize
from datetime import timedelta
#import plotly.graph_objects as go
import pytz
import warnings
import matplotlib.pyplot as plt
import math
from keras.models import Model, Sequential
import folium
import holidays
import boto3
import pyspark.sql.functions as F
import pickle

## Modeling Libraries

# Model Evaluation
from sklearn.metrics import mean_squared_error as mse

# ARIMA/VAR Models
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from keras.models import Model, Sequential

# Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from fbprophet import Prophet

# Saving plots
import io

import warnings
warnings.filterwarnings("ignore")

## Connect to AWS
secret_scope = "w210-scope"
access_key = dbutils.secrets.get(scope=secret_scope, key="aws-access-key")
secret_key = dbutils.secrets.get(scope=secret_scope, key="aws-secret-key")
encoded_secret_key = secret_key.replace("/", "%2F")
aws_bucket_name = "w210v2"
mount_name = "w210v2"

try:
    dbutils.fs.mount("s3a://%s:%s@%s" % (access_key, encoded_secret_key, aws_bucket_name), "/mnt/%s" % mount_name)
except Exception as e:
    print("Already mounted :)")

# COMMAND ----------

# MAGIC %md
# MAGIC # Reference Functions

# COMMAND ----------

# MAGIC %md
# MAGIC 1. `arima_filter` -- filter time series dataframe based on start and end dates
# MAGIC 2. `dfSplit_Xy` -- prepare the dataframe for LSTM model by getting the X input and y output as np arrays
# MAGIC 3. `split_train_test` -- split the data into training data and development/validation data (14 days)
# MAGIC 4. `run_lstm` -- create the LSTM model
# MAGIC 5. `write_model_to_s3` -- write the LSTM model using pickle to S3 bucket
# MAGIC 6. `get_model_from_s3` -- read the LSTM model in pickle from S3 bucket

# COMMAND ----------

def arima_filter(df, start_date, end_date, date_col):
    ''' 
    filter data frame to be between the start date and end date, not including the end date. 
    date_col is the date column used to filter the dataframe
    '''
    return df[(df[date_col] >= start_date) & (df[date_col] < end_date)] # should we reset the index

  
def dfSplit_Xy(df, date_col='datetime', n_input=6, n_out=36):
    """ 
    Tranform pandas dataframe into arrays of inputs and outputs. 
    The output (value predicting) must be located at the end of the df
    n_inputs, is the the number of inputs, to use to predict n_input+1 (or i + n_input) aka window size
    n_outs, is the number of steps out you want to predict. Default is 1
    Returns 2 numpy arrays. Inputs (features), and actual labels
    """
    
    ind_dates = df[date_col].to_numpy() ####
    df_as_np = df.set_index(date_col).to_numpy()
    
    #X is matrix of inputs
    #y is actual outputs, labels
    X = []
    y = []
    y_dates = [] ####
    
    
    for i in range(len(df_as_np)): 
        #print(i)
        start_out = i + n_input
        start_end = start_out + n_out

        # to make sure we always have n_out values in label array
        if start_end > len(df_as_np):
            break

        #take the i and the next values, makes a list of list for multiple inputs
        row = df_as_np[i:start_out, :]
        #print(row)
        X.append(row)

        # Creating label outputs extended n_out steps. -1, last column is label
        label = df_as_np[start_out:start_end, -1]
        #print(label)
        y.append(label)
        
        # array of dates
        label_dates = ind_dates[start_out:start_end]####
        #print(label_dates)###
        y_dates.append(label_dates) #### 
        
#         print('X shape == {}.'.format(np.array(X).shape))
#         print('y shape == {}.'.format(np.array(y).shape))
    # can we have a ydates for the dates??? and timesteps
    
    return np.array(X), np.array(y), np.array(y_dates)


def split_train_test(X, y):
    '''
    Split inputs and labels into train, validation and test sets for LSTMs
    Returns x and y arrays for train, validation, and test.
    '''
    
    dev_prop, train_prop = 14/90, 1 - (14/90)
    
    # get size of Array
    num_timestamps = X.shape[0]
    
    # define split points
    train_start = 0 # do we need this? can we always start at 0?
    train_end = int(num_timestamps * train_prop)
    dev_end = int(num_timestamps * (train_prop + dev_prop))
    
    # splitting
    X_train, y_train = X[train_start:train_end], y[train_start:train_end]#, y_dates[train_start:train_end]
    X_val, y_val = X[train_end:dev_end], y[train_end:dev_end]#, y_dates[train_end:dev_end]
    # include dates for plotting later
    # print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_train.shape[0])
    
    return X_train, y_train, X_val, y_val


def run_lstm(n_inputs, n_features, n_outputs, X_train, y_train, X_val, y_val, n_epochs = 10):
    '''Run lstm model, and get fitted model'''
    
    # Build LSTM Model
    model = Sequential()
    model.add(InputLayer((n_inputs, n_features))) 
    model.add(LSTM(64, input_shape = (n_inputs, n_features)))
    model.add(Dense(8, 'relu'))
    model.add(Dense(n_outputs, 'linear'))
    model.summary()
    
    ### checkpoints to save best model
    cp = ModelCheckpoint('model/', save_best_only=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    
    ## Fit model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, callbacks=[cp])
    
    # load model in the event the last epoch is not the best model, if it is will give same results if skip
    model = load_model('model/')
    
    return model
  
    
def write_model_to_s3(model, model_name):
    
    bucket='w210v2'
    key=f'models/{model_name}.pkl'
    
    client = boto3.client("s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    pickle_byte_obj = pickle.dumps(model) 

    client.put_object(Key=key, Body=pickle_byte_obj, Bucket="w210v2")
    
    
def write_seed_model_to_s3(model, model_name):
    
    bucket='w210v2'
    key=f'models/seed_models/{model_name}.pkl'
    
    client = boto3.client("s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    pickle_byte_obj = pickle.dumps(model) 

    client.put_object(Key=key, Body=pickle_byte_obj, Bucket="w210v2")
    
    
def get_model_from_s3(model_name):
    
    bucket='w210v2'
    key=f'models/{model_name}.pkl'

    client = boto3.client("s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    response = client.get_object(Bucket=bucket,Key=key)

    model = pickle.loads(response['Body'].read())
    return model

# COMMAND ----------

def save_fig_to_s3(fig_name):
    bucket='w210v2'
    key=f'figures/{fig_name}.png'
    
    client = boto3.client("s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    plt.plot(range(10), range(10))
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    client.put_object(Key=key, Body=img_data, Bucket="w210v2", ContentType = 'image/png')

# COMMAND ----------

# DBTITLE 1,Write to S3 Buckets
# write results to s3 bucket

def write_6m_to_s3(df, save_filename):

    # create spark data frame
    results_ab = spark.createDataFrame(df) 

    ## Write to AWS S3
    (results_ab
        .repartition(1)
        .write
        .format("parquet")
        .mode("overwrite")
        .save(f"/mnt/{mount_name}/data/{save_filename}"))

# COMMAND ----------

# write results to s3 bucket

def write_seed_to_s3(df, save_filename):

    # create spark data frame
    results_ab = spark.createDataFrame(df) 

    ## Write to AWS S3
    (results_ab
        .repartition(1)
        .write
        .format("parquet")
        .mode("overwrite")
        .save(f"/mnt/{mount_name}/data/{save_filename}"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Seed Datasets

# COMMAND ----------

# DBTITLE 1,Load Seed Datasets
slrp_seed = spark.read.parquet(f"/mnt/{mount_name}/data/slrp_seed")
slrp_seed = slrp_seed.toPandas()
print("Loaded Slrp Seed")

# COMMAND ----------

palo_alto_seed = spark.read.parquet(f"/mnt/{mount_name}/data/palo_alto_seed")
palo_alto_seed = palo_alto_seed.toPandas()
print("Loaded Palo Alto Seed")

# COMMAND ----------

boulder_seed = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_seed")
boulder_seed = boulder_seed.toPandas()
print("Loaded Boulder Seed")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load 6-Month Datasets

# COMMAND ----------

slrp_6m = spark.read.parquet(f"/mnt/{mount_name}/data/slrp_6m")
slrp_6m = slrp_6m.toPandas()
print("Loaded Slrp 6M")

# COMMAND ----------

palo_alto_6m = spark.read.parquet(f"/mnt/{mount_name}/data/palo_alto_6m")
palo_alto_6m = palo_alto_6m.toPandas()
print("Loaded Palo Alto 6M")

# COMMAND ----------

boulder_6m = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_6m")
boulder_6m = boulder_6m.toPandas()
print("Loaded Boulder 6M")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Streaming Data

# COMMAND ----------

slrp_stream = spark.read.parquet(f"/mnt/{mount_name}/data/slrp_stream")
slrp_stream = slrp_stream.toPandas()
print("Loaded Slrp Streaming")

palo_alto_stream = spark.read.parquet(f"/mnt/{mount_name}/data/palo_alto_stream")
palo_alto_stream = palo_alto_stream.toPandas()
print("Loaded Palo Alto Streaming")

boulder_stream = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_stream")
boulder_stream = boulder_stream.toPandas()
print("Loaded Boulder Streaming")

# COMMAND ----------

# MAGIC %md
# MAGIC # LSTM Model on Seed Data

# COMMAND ----------

## Define timestep constants
n_inputs = 12
n_outputs = 36

date_col="datetime"
station_col = "station"

models_dir_entries = dbutils.fs.ls(f"/mnt/{mount_name}/models/")
seed_model_dir_entries = dbutils.fs.ls(f"/mnt/{mount_name}/models/seed_models/")

## If there is more than just the "seed_models" folder in the "models" directory
if len(models_dir_entries) > 1:
    
    ## Loop through all filesystem objects within the models directory
    for models_dir_entry in models_dir_entries:
        
        ## Delete existing model files
        if models_dir_entry.isFile():
            try:
                dbutils.fs.rm(models_dir_entry.path)
            except:
                print(f"Couldn't delete: {models_dir_entry.path}")
        
    ## Copy seed models back into directory
    for seed_model_dir_entry in seed_model_dir_entries:
        dbutils.fs.cp(seed_model_dir_entry.path, f"/mnt/{mount_name}/models/")

        
def train_seed_models(location_df):
    
    ## Define the time window for our predictions to be based on
#     end_date = location_df[date_col].max()
#     start_date = end - dt.timedelta(days = int(3*30))
    
#     ## Filter the dataframe to the desired timeframe
#     location_df = arima_filter(location_df, 
#                                start_date=start_date, 
#                                end_date=end_date, 
#                                date_col=date_col)

    ## Get a list of all the unique stations 
    stations = list(location_df[station_col].unique())

    ## Iterate through every station in this region
    for station in stations:

        ## Filter to just the station at hand
        station_df = location_df[location_df[station_col] == station]
        station_df = station_df.drop(station_col, axis=1)
        
        ## Split the data
#         global X
        X, y, dates = dfSplit_Xy(station_df, date_col="datetime", n_input=n_inputs, n_out=n_outputs)
        print(f"1st: {X}, {y}, {dates}")
        
        ## Split the data into train, validation, (and test)
#         global X_val
        X_train, y_train, X_val, y_val = split_train_test(X, y)
        print(f"2nd: {X_train}, {y_train}, {X_val}, {y_val}")
        
        ## Fit the model
        model = run_lstm(n_inputs=n_inputs, 
                         n_features=X_train.shape[2],
                         n_outputs=n_outputs, 
                         X_train=X_train,
                         y_train=y_train,
                         X_val=X_val, 
                         y_val=y_val, 
                         n_epochs = 10)
        
        ## Write the model to s3
        print("writing to s3...")
        write_seed_model_to_s3(model, station + "_model")
        
        
def train_models(station_df):
    
    ## Define the time window for our predictions to be based on
#     end_date = location_df[date_col].max()
#     start_date = end - dt.timedelta(days = int(3*30))
    
#     ## Filter the dataframe to the desired timeframe
#     location_df = arima_filter(location_df, 
#                                start_date=start_date, 
#                                end_date=end_date, 
#                                date_col=date_col)

    ## Get a list of all the unique stations 
    stations = list(station_df[station_col].unique())
    if len(stations) > 1:
        return "Too many stations."
    station_name = stations[0]

    station_df = station_df.drop(station_col, axis=1)

    ## Split the data
#         global X
    X, y, dates = dfSplit_Xy(station_df, date_col="datetime", n_input=n_inputs, n_out=n_outputs)
    # print(f"1st: {X}, {y}, {dates}")

    ## Split the data into train, validation, (and test)
#         global X_val
    X_train, y_train, X_val, y_val = split_train_test(X, y)
    # print(f"2nd: {X_train}, {y_train}, {X_val}, {y_val}")

    ## Fit the model
    model = run_lstm(n_inputs=n_inputs, 
                     n_features=X_train.shape[2],
                     n_outputs=n_outputs, 
                     X_train=X_train,
                     y_train=y_train,
                     X_val=X_val, 
                     y_val=y_val, 
                     n_epochs = 10)

    ## Write the model to s3
    print("writing to s3...")
    write_model_to_s3(model, station_name + "_model")

# COMMAND ----------

# DBTITLE 1,Train Seed Models
if False:
    train_seed_models(slrp_seed)
    train_seed_models(palo_alto_seed)
    train_seed_models(boulder_seed)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Make Predictions from LSTM Model

# COMMAND ----------

def make_predictions(station, location_df, streamed_df, location_df_6m):
    """
    model_name: name of model you want to load from S3
    location_df: training data
    location_df_6m: all data (training + streaming)
    
    Make predictions from an LSTM model by getting the station information from the last `n_inputs` timestamps
    and running it through the model for predictions
    
    Use the location_df_6m dataframe to calculate the errors (MSE, RMSE) of the predictions
    
    This function assumes that we are only predicting using a single feature (no time-based features)
    """
    
    model_name = station + "_model"
    model = get_model_from_s3(model_name)
    
    # For testing, concatenate the original dataframe (seed) with the sreamed data
    full_historical = pd.concat([location_df, streamed_df]).sort_values(by = 'datetime')
    
    ## COMMENTING OUT BECAUSE WE PLAN TO MAKE FIGURES IN TABLEAU, NOT THROUGH MATPLOTLIB
    # Setup for saving figures
#     bucket='w210v2'
#     client = boto3.client("s3",
#         aws_access_key_id=access_key,
#         aws_secret_access_key=secret_key,
#     )
    
    
    # Loop through each unique station
#     for station in location_df['station'].unique():
        
    # Filter to that station's values
    station_historical = full_historical[full_historical['station'] == station].drop(columns = ['station'])
    location_df_6m = location_df_6m[location_df_6m["station"] == station]

    # Develop a testing point to predict out n_output time steps
    test = np.array(station_historical.set_index('datetime', drop = True).tail(n_inputs))
    test.resize(1, n_inputs, test.shape[1]) # 1 record, number of timesteps back (can also be test.shape[0], number of features

    # Make predictions from the model
    predictions = model.predict(test)[0]
    predictions_rounded = np.round(predictions)

    # Get the actual values from the 6 month data to calculate the MSE
    actuals = location_df_6m[(location_df_6m['datetime'] > station_historical['datetime'].max()) &
                             (location_df_6m['datetime'] <= station_historical['datetime'].max() + 
                              dt.timedelta(minutes = 10*n_outputs))]

    # Add the predictions to this data frame
    actuals['predicted'] = predictions
    actuals['predicted_rounded'] = predictions_rounded
    actuals = actuals.reset_index(drop = True)

    # For sanity checking, print the maximum timestamp from the historical data and the minimum value from the predictions
    # print(full_historical['datetime'].max())
    # print(actuals['datetime'].min())

    ## Evaluation Metrics ###
    MSE_raw = mse(actuals['ports_available'], actuals['predicted'])
    MSE_rounded = mse(actuals['ports_available'], actuals['predicted_rounded'])
    RMSE_raw = math.sqrt(MSE_raw)
    RMSE_rounded = math.sqrt(MSE_rounded)

    ### CAN COMMENT THIS OUT
    actuals['abs_error'] = (actuals['ports_available'] - actuals['predicted_rounded']).abs()

    ### COMMENTING OUT SINCE WE WANT ERRORS ON AN INDIVIDUAL TIMESTEP LEVEL, WHICH WE CAN THEN AGGREGATE TO HOURLY
#         # Calcualte RMSEs for each hour
#         RMSE_1 = np.sqrt(mse(actuals.iloc[0:6, :]['ports_available'], actuals.iloc[0:6,:]['predicted_rounded']))
#         RMSE_2 = np.sqrt(mse(actuals.iloc[6:12, :]['ports_available'], actuals.iloc[6:12,:]['predicted_rounded']))
#         RMSE_3 = np.sqrt(mse(actuals.iloc[12:18, :]['ports_available'], actuals.iloc[12:18,:]['predicted_rounded']))
#         RMSE_4 = np.sqrt(mse(actuals.iloc[18:24, :]['ports_available'], actuals.iloc[18:24,:]['predicted_rounded']))
#         RMSE_5 = np.sqrt(mse(actuals.iloc[24:30, :]['ports_available'], actuals.iloc[24:30,:]['predicted_rounded']))
#         RMSE_6 = np.sqrt(mse(actuals.iloc[30:36, :]['ports_available'], actuals.iloc[30:36,:]['predicted_rounded']))

#         new_results = pd.DataFrame(data = {'Overall RMSE': [RMSE_rounded],
#                                            'RMSE for Hour 1': [RMSE_1],
#                                            'RMSE for Hour 2': [RMSE_2],
#                                            'RMSE for Hour 3': [RMSE_3],
#                                            'RMSE for Hour 4': [RMSE_4],
#                                            'RMSE for Hour 5': [RMSE_5],
#                                            'RMSE for Hour 6': [RMSE_6]},
#                                    index = [actuals['datetime'].min()])

#         print(RMSE_raw)
#         print(RMSE_rounded)

    # Save as a single dataframe n_outputs number of historical data, n_outputs actuals, and n_outputs predictions
    tableau_plot_data = pd.DataFrame(data = {'datetime': station_historical.tail(n_outputs + streamed_df.shape[0])['datetime'],
                                             'station': [station]*(n_outputs + streamed_df.shape[0]),
                                             'ports_available': station_historical.tail(n_outputs + streamed_df.shape[0])['ports_available'],
                                             'label': ['historical']*(n_outputs + streamed_df.shape[0]),
                                             'stream_count': streamed_df.shape[0]})
    tableau_plot_data = pd.concat([tableau_plot_data, 
                                   pd.DataFrame(data = {'datetime': actuals['datetime'],
                                             'station': [station]*n_outputs,
                                             'ports_available': actuals['ports_available'],
                                             'label': ['actual']*n_outputs,
                                             'stream_count': streamed_df.shape[0]})], 
                                  ignore_index = True)
    tableau_plot_data = pd.concat([tableau_plot_data, 
                                   pd.DataFrame(data = {'datetime': actuals['datetime'],
                                             'station': [station]*n_outputs,
                                             'ports_available': actuals['predicted_rounded'],
                                             'label': ['predicted']*n_outputs,
                                             'stream_count': streamed_df.shape[0]})], 
                                  ignore_index = True)

    tableau_plot_data_save = spark.createDataFrame(tableau_plot_data)
    ## Write to AWS S3
    (tableau_plot_data_save
        .write
        .format("parquet")
        .mode("append")
        .save(f"/mnt/{mount_name}/data/streaming_predictions/{station.lower()}_preds"))

    timestep_errors = pd.DataFrame(data = {'datetime': actuals['datetime'],
                                           'station': [station]*n_outputs,
                                           'ports_available': actuals['ports_available'],
                                           'predicted_rounded': actuals['predicted_rounded'],
                                           'error': actuals['abs_error'],
                                           'stream_count': streamed_df.shape[0],
                                           'timesteps_out': range(n_outputs)})
    timestep_errors_save = spark.createDataFrame(timestep_errors)
    ## Write to AWS S3
    (timestep_errors_save
        .write
        .format("parquet")
        .mode("append")
        .save(f"/mnt/{mount_name}/data/streaming_predictions/{station.lower()}_pred_errors"))

#         # plot actuals and predictions
#         plt.subplots(figsize = (8,6))
#         plt.plot(station_historical.tail(n_outputs)['datetime'], 
#                  station_historical.tail(n_outputs)['ports_available'], label = 'Historical')
#         plt.plot(actuals['datetime'], actuals['ports_available'], label = 'Actual')
#         plt.plot(actuals['datetime'], predictions_rounded, '--', label = 'Predictions')
#     #     plt.plot(df[date_col], df[output_col], label = 'Actuals')

#         plt.ylim(0, location_df_6m['ports_available'].max() * 1.1)

#         plt.xlabel('DateTime', fontsize = 16);
#         plt.ylabel('Number of Available Stations', fontsize = 16);
#         plt.legend(fontsize = 14);
#         plt.title('Charging Station Availability for ' + station, fontsize = 18);

#         img_data = io.BytesIO()
#         plt.savefig(img_data, format='png')
#         img_data.seek(0)
#         fig_name = station + '_' + str(actuals['datetime'].min())
#         key=f'figures/{fig_name}.png'
#         client.put_object(Key=key, Body=img_data, Bucket="w210v2", ContentType = 'image/png')

#         return pd.concat([results, new_results])
    return tableau_plot_data