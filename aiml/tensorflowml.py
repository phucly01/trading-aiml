from cgi import test
from matplotlib import units
import pandas as pd
import numpy as np
import numba as nb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
import collections
from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import matplotlib.pyplot as plot
import datetime
from mplcursors import cursor
class TensorFlow:
    
    def __init__(self, 
                 symbol: str, 
                 df: pd.DataFrame, 
                 index: str, 
                 targets: list=["close", "open", "high", "low"], 
                 predict_steps: int=15, 
                 batch_size: int=50, 
                 epoch: int=10,
                 test_size: int=10):
        self.index = index
        self.symbol = symbol
        self.targets = targets
        self.predict_steps = predict_steps
        self.batch_size = batch_size
        self.epoch = epoch
        self.test_size = test_size
        self.df = df.copy()
        
        
    def load_data(self, df, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                test_size=0.2, feature_columns=['close', 'volume', 'open', 'high', 'low']):
        """
        Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
        Params:
            ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
            n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
            scale (bool): whether to scale prices from 0 to 1, default is True
            shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
            lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
            split_by_date (bool): whether we split the dataset into training/testing by date, setting it 
                to False will split datasets in a random way
            test_size (float): ratio for test data, default is 0.2 (20% testing data)
            feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
        """
        # see if ticker is already a loaded stock from yahoo finance
        type(df)
        # this will contain all the elements we want to return from this function
        result = {}
        # we will also return the original dataframe itself
        result['df'] = df.copy()

        # make sure that the passed feature_columns exist in the dataframe
        for col in feature_columns:
            assert col in df.columns, f"'{col}' does not exist in the dataframe."

        # add date as a column
        if "date" not in df.columns:
            df["date"] = df.index

        if scale:
            column_scaler = {}
            # scale the data (prices) from 0 to 1
            for column in feature_columns:
                scaler = preprocessing.MinMaxScaler()
                df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
                column_scaler[column] = scaler

            # add the MinMaxScaler instances to the result returned
            result["column_scaler"] = column_scaler

        # add the target column (label) by shifting by `lookup_step`
        df['future'] = df['close'].shift(-lookup_step)

        # last `lookup_step` columns contains NaN in future column
        # get them before droping NaNs
        last_sequence = np.array(df[feature_columns].tail(lookup_step))
        
        # drop NaNs
        df.dropna(inplace=True)

        sequence_data = []
        sequences = collections.deque(maxlen=n_steps)

        for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])

        # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
        # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
        # this last_sequence will be used to predict future stock prices that are not available in the dataset
        last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
        last_sequence = np.array(last_sequence).astype(np.float32)
        # add to result
        result['last_sequence'] = last_sequence
        
        # construct the X's and y's
        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)

        # convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        if split_by_date:
            # split the dataset into training & testing sets by date (not randomly splitting)
            train_samples = int((1 - test_size) * len(X))
            result["X_train"] = X[:train_samples]
            result["y_train"] = y[:train_samples]
            result["X_test"]  = X[train_samples:]
            result["y_test"]  = y[train_samples:]
            if shuffle:
                # shuffle the datasets for training (if shuffle parameter is set)
                shuffle_in_unison(result["X_train"], result["y_train"])
                shuffle_in_unison(result["X_test"], result["y_test"])
        else:    
            # split the dataset randomly
            result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                    test_size=test_size, shuffle=shuffle)

        # get the list of test set dates
        dates = result["X_test"][:, -1, -1]
        # retrieve test features from the original dataframe
        result["test_df"] = result["df"].loc[dates]
        # remove duplicated dates in the testing dataframe
        result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
        # remove dates from the training/testing sets & convert to float32
        result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

        return result
        
    def prepare_data(self, predict_target):
        self.predict_target = predict_target
        self.adj_data = {}
        self.scalers = {}
        
        for col in self.targets:
            self.scalers[col] = preprocessing.MinMaxScaler()
            self.df[col] = self.scalers[col].fit_transform(np.expand_dims(self.df[col].values, axis=1))
            
        futurecol = "new"+self.predict_target
        self.df[futurecol] = self.df[self.predict_target].roll(-self.predict_steps)
        
        final_batch = self.df[self.targets].tail(self.predict_steps)
        
        self.df.dropna(inplact=True)
        batches = []
        batch = collections.deque(maxlen=self.batch_size)
        y = []
        
        for data, expected in zip(self.df[[self.index] + self.targets].values, self.df[futurecol].values):
            batch.append(data)
            if batch.count() == batch.maxlen:
                batches.append(np.array(batch))
                y.append(expected)
        final_batch = np.array(list([b[:len(self.targets)] for b in batches]) + list(final_batch))
        
        X = np.array(batches)
         
        
         
class TensorFlowMLAdam:

    def __init__(self, symbol:str, df: pd.DataFrame, index: str, data: str, future_steps: int=15, sampling_size: int=50, test_size: int=10, iterations: int=200) -> None:
        """
        df - the dataframe containing index and data
        index - the name of the index list
        data - the list of names of the data lists
        predict_step - the number of future steps to predict
        """
        self.index_data = df[[index]].values
        self.index_type = df[index].dtypes.name
        self.index = index
        self.column_data = df[[data]].values
        self.column = data
 
        self.symbol = symbol 
        self.future_steps = future_steps
        self.sampling_size = sampling_size 
        self.test_size = test_size
        self.iterations = iterations 
        self.scaler = None
        self.data_scaler = None
        
        self.prepare_data()

    def create_model(self, data):
        model = Sequential()
        
        model.add(LSTM(units=256, return_sequences=True, activation='relu', input_shape=(data.shape[1], data.shape[2])))
        model.add(LSTM(units=128, return_sequences=True, activation='relu', input_shape=(data.shape[1], data.shape[2])))
        model.add(LSTM(units=64, return_sequences=False, activation='relu')) 
        model.add(Dense(units=1))
        self.model = model
        

    def train(self):     
        self.create_model(self.train_x)
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
        history = self.model.fit(self.train_x, self.train_y, epochs=self.iterations)
        

    def add_data(index, data) -> int:
        pass

    def prepare_data(self):        
        X = list()
        y = list()
         
        if self.scaler is None:
            sc = preprocessing.MinMaxScaler()
            self.scaler = sc
        
        self.data_scaler = self.scaler.fit(self.column_data)
         
        data = self.data_scaler.transform(self.column_data)
        
        #print(self.column_data[0:2], data[0:2])

        for i in range(self.sampling_size, len(data)):
            X.append(data[i-self.sampling_size:i])
            y.append(data[i])
        
        self.inputs = np.array(X)
        self.inputs = self.inputs.reshape(self.inputs.shape[0], self.inputs.shape[1], 1)
        self.target = np.array(y)
        
        self.training_size = int((100 - self.test_size)/100 * len(data))
        
        self.train_x = self.inputs[:self.training_size]
        self.train_y = self.target[:self.training_size]
        self.test_x = self.inputs[self.training_size:]
        self.test_y = self.target[self.training_size:]   
        
        print("Train shapes: x={}, y={}".format(self.train_x.shape, self.train_y.shape))
        print("Test shapes: x={}, y={}".format(self.test_x.shape, self.test_y.shape))
        

    def test(self): 
        results = self.model.evaluate(self.test_x, self.test_y, verbose=1)
        print(results) 
        predicted = self.model.predict(self.test_x)
        accuracy = 100 - (100 * (abs(self.test_y - predicted)/self.test_y)).mean()
        print("Accuracy: {}".format(accuracy))

    def predict_next(self, data):
        if data is None:
            data = self.column_data[-self.sampling_size:]
        last_batch = data
        
        last_batch = self.data_scaler.transform(last_batch.reshape(-1,1))
        last_batch = last_batch.reshape(1, self.sampling_size, 1)
        predicted = self.model.predict(last_batch)
        predicted_price = self.data_scaler.inverse_transform(predicted)
        print("{} -> {}".format(predicted, predicted_price)) 
        data = np.append(data[1:], [round(predicted_price[0][0], 2)])
        return data
    
    def predict(self):
        data = None
        for i in range(self.future_steps):
            data = self.predict_next(data)
        self.predictions = data
            

    def plot(self, days):
        past_index = days - self.future_steps if days > self.future_steps else self.future_steps
        data = None
        index = None
        if days <= self.future_steps:
            data = self.column_data[-1:].flatten()
            index = self.index_data[-1:].flatten()
        else:
            data = self.column_data[-past_index:].flatten()
            index = self.index_data[-past_index:].flatten()
        previous = str(index[-1:][0])
        
        previous = datetime.date.fromisoformat(previous)
        
        #Create a list of dates skipping Sat and Sun
        day=1
        future_index = [index[-1:][0]]
        end = self.future_steps
        while day <= end:
            tomorrow = previous + datetime.timedelta(days=day)
            if tomorrow.weekday() in [0,1,2,3,4]:
                future_index.append(tomorrow.isoformat())
            else:
                end += 1
            day += 1
         
        future_data = self.predictions[-len(future_index):]
        # previous = [(previous + datetime.timedelta(days=i)).isoformat() for i in range(self.future_steps)]  #Create a list dates starting from yesterday
        
        fig = plot.figure(figsize=(10, 8))
        fig.autofmt_xdate(rotation=45)
        plot.plot(index, data)
        plot.plot(future_index, future_data)
        plot.grid(visible=True, axis='both', which='both')
        plot.title('{} Stock Close Price'.format(self.symbol))
        plot.xlabel('Date')
        plot.ylabel("Close")
        plot.legend(['Past', 'Predictions'])
        cursor(hover=True)
        plot.show()
         