# -*- coding: utf-8 -*-
"""
Author:
    lprtk

Description:
    It is a Python library for time series feature selection. The objective is 
    to focus on building a "good" model. To define a "good" model, we rely on
    Theil's metrics which allow us to conclude on the goodness of fit of the
    predictions made by a model. This library allows you to find the best model 
    you need with based on your desired criteria.

License:
    MIT License
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import choice, randint
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted


#------------------------------------------------------------------------------


class Metrics:
    def __init__(self, y_true, y_pred) -> None:
        """
        Class allows to implement all several metrics use to measure the predictive
        capabilities of a regression model in and out-of-sample.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Ground truth (correct) target values.
            
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Estimated target values.

        Raises
        ------
        TypeError
            - y_true parameter must be a ndarray or series to use the functions
            associated with the Metrics class.
            
            - y_pred parameter must be a ndarray or series to use the functions
            associated with the Metrics class.
            
        ValueError
            y_true and y_pred must have the same shape to use the functions associated
            with the Metrics class.

        Returns
        -------
        None.

        """
        if y_true.shape[0] == y_pred.shape[0]:
            
            if isinstance(y_true, pd.core.series.Series) or isinstance(y_true, np.ndarray):
                self.y_true = y_true
            else:
                raise TypeError(
                    f"'y_true' parameter must be a ndarray or series: got {type(y_true)}"
                )

            if isinstance(y_pred, pd.core.series.Series) or isinstance(y_pred, np.ndarray):
                self.y_pred = y_pred
            else:
                raise TypeError(
                    f"'y_pred' parameter must be a ndarray or series: got {type(y_pred)}"
                )
                
        
        else:
            raise ValueError(
                "'y_true' and 'y_pred' parameters must have the same shape"
                )
    
    
    def mse(self) -> float:
        """
        Function to compute the Mean Squared Error (MSE) regression loss.

        Returns
        -------
        float
            A non-negative floating point value (the best value is 0.0) which corresponds
            to the loss value of the MSE.

        """
        mse = ((1/self.y_pred.shape[0]) * sum((np.array(self.y_true)\
                                               -np.array(self.y_pred))**2))
        
        return mse
    
    
    def rmse(self) -> float:
        """
        Function to compute the Root Mean Squared Error (RMSE) regression loss.

        Returns
        -------
        float
            A non-negative floating point value (the best value is 0.0) which corresponds
            to the loss value of the RMSE.

        """
        rmse = np.sqrt((1/self.y_pred.shape[0]) * sum((np.array(self.y_true)\
                                                       -np.array(self.y_pred))**2))
        
        return rmse
    
    
    def mae(self) -> float:
        """
        Function to compute the Mean Absolute Error (MAE) regression loss.

        Returns
        -------
        float
            A non-negative floating point value (the best value is 0.0) which corresponds
            to the loss value of the MAE.

        """
        mae = ((1/self.y_pred.shape[0]) * sum(np.abs(self.y_true-self.y_pred)))
        
        return mae
    
    
    def mape(self) -> float:
        """
        Function to compute the Mean Absolute Percentage Error (MAPE) regression loss.

        Returns
        -------
        float
            A non-negative floating point value (the best value is 0.0) which corresponds
            to the loss value of the MAPE.

        """
        mape = ((100/self.y_pred.shape[0]) * np.sum(np.abs((self.y_true-self.y_pred)/self.y_true)))
        
        return float(mape)
    
    
    def um_theil(self) -> float: 
        """
        Function to compute the Theil UM (bias) metric. Bias (UM) is an indication of
        systematic error and measures the extent to which the average values of the
        actual and predicted deviate from each other.

        Returns
        -------
        float
            A non-negative floating point value (the best value is 0.0) which corresponds
            to the bias (UM) calculated.

        """
        numerator = ((np.mean(self.y_true)-np.mean(self.y_pred))**2)
        denominator = ((1/self.y_pred.shape[0]) * sum((np.array(self.y_true)\
                                                       -np.array(self.y_pred))**2))
        
        if denominator == 0:
            um = 0
        else:
            um = numerator / denominator
            
        return um
    
    
    def us_theil(self) -> float:
        """
        Function to compute the Theil US (var) metric, where var (US) is the variance
        proportion. US indicates the ability of the model to replicate the degree of
        variability in the endogenous variable.

        Returns
        -------
        float
            A non-negative floating point value (the best value is 0.0) which corresponds
            to the var (US) calculated.

        """
        numerator = ((np.std(self.y_pred)-np.std(self.y_true))**2)
        denominator = ((1/self.y_pred.shape[0]) * sum((np.array(self.y_true)\
                                                       -np.array(self.y_pred))**2))
        
        if denominator == 0:
            us = 0
        else:
            us = numerator / denominator
        
        return us
    
    
    def uc_theil(self) -> float:
        """
        Function to compute the Theil UC (covar) metric. Covar represents the remaining
        error after deviations from average values and average variabilities have been
        accounted for.

        Returns
        -------
        float
            A non-negative floating point value (the best value is 0.0) which corresponds
            to the covar (UC) calculated.

        """
        numerator = (2*(1-((np.cov(self.y_true, self.y_pred)[0][1])\
                           /(np.std(self.y_true)*np.std(self.y_pred))))*\
                     np.std(self.y_pred)*np.std(self.y_true))
        denominator = ((1/self.y_pred.shape[0]) * sum((np.array(self.y_true)\
                                                       -np.array(self.y_pred))**2))
        
        if denominator == 0:
            uc = 0
        else:
            uc = numerator / denominator
        
        return uc
    
    
    def u1_theil(self) -> float:
        """
        Function to compute the Theil U1 metric. U1 is a statistic that measures the
        accuracy of a forecast.

        Returns
        -------
        float
            A non-negative floating point value (the best value is 0.0) which corresponds
            to the U1 statistic calculated.

        """
        numerator = np.sqrt((1/self.y_pred.shape[0]) * sum((np.array(self.y_true)\
                                                            -np.array(self.y_pred))**2))
        denominator = np.sqrt((1/self.y_pred.shape[0]) * sum(np.array(self.y_true)**2))
        
        if denominator == 0:
            u1 = 0
        else:
            u1 = numerator / denominator
        
        return u1
    
    
    def u_theil(self) -> float:
        """
        Function to compute the Theil U metric. U is the Theil’s inequality coefficient.

        Returns
        -------
        float
            A non-negative floating point value (the best value is 0.0) which corresponds
            to the U statistic calculated.

        """
        numerator = np.sqrt((1/self.y_pred.shape[0]) * sum((np.array(self.y_true)\
                                                            -np.array(self.y_pred))**2))
        denominator = np.sqrt((1/self.y_pred.shape[0])*sum(np.array(self.y_true)**2))\
            + np.sqrt((1/self.y_pred.shape[0])*sum(np.array(self.y_pred)**2))
        
        if denominator == 0:
            u = 0
        else:
            u = numerator / denominator
        
        return u


#------------------------------------------------------------------------------


class FeatureSelection:
    def __init__(self, X_train, X_test, y_train, y_test, scoring: str="U",
                 n_iter: int=100, min_features=None, y_lag_select=None,
                 random_state: int=42) -> None:
        """
        Class allows to implement a features selection based on Theil metrics 
        (UM, US, UC, U1, U). This selection technique estimates combinations of
        sub-models (like backward, forward, stepwise selection or bayesian moving
        averaging) in order to find the combination of variables (e.g. the model)
        that minimizes a given criterion. In other words, this allows to find the
        model (e.g. the features) that minimizes the estimation errors on the in
        and out-of-sample points, thus the best predictive capacities and the best
        quality of fit.

        Parameters
        ----------
        X_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Matrix design values ont the train set (in-sample predictions).
            
        X_test : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Matrix design values ont the test set (out-of-sample predictions).
            
        y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values on the train set (in-sample predictions).
            
        y_test : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values on the test set (out-of-sample predictions).
            
        scoring : {"UM", "US", "UC", "U1", "U"}, str, optional, default="U"
            Metric to minimize or maximize in order to find the best model.
            Default is "U".
            
        n_iter : int, optional, default=100
            Number of iterations corresponding to the number of estimated sub-models.
            Default is 100.
            
        min_features : None or int, optional, default=None
            If random_select=True, this specifies the minimum number of variables 
            in each sub-model. Default is None. If random_select=False,
            then min_features must be None.
            
        y_lag_select : None or list, optional, default=None
            If there are lagged endogenous features in the matrix design, the
            list contains the lag's names of the feature to be explained that
            must be in the model. Default is None.
            Example of use: y_lag_select=["y_lag1", "y_lga2"]
            
        random_state : int, optional, default=42
            Controls the randomness of estimations. Default is 42.

        Raises
        ------
        TypeError
            - X_train parameter must be a ndarray or dataframe to use the
            functions associated with the FeatureSelection class.
            
            - X_test parameter must be a ndarray or dataframe to use the
            functions associated with the FeatureSelection class.
            
            - y_train parameter must be a ndarray or series to use the
            functions associated with the FeatureSelection class.
            
            - y_test parameter must be a ndarray or series to use the
            functions associated with the FeatureSelection class.
            
            - scoring parameter must be a str to use the functions associated
            with the FeatureSelection class.
            
            - n_iter parameter must be an int to use the functions associated
            with the FeatureSelection class.
            
            - min_features parameter must be an int or None to use the functions
            associated with  the FeatureSelection class.
            
            - y_lag_select parameter must be a list or None to use the functions
            associated with the FeatureSelection class.
            
            - random_state parameter must be an int to use the functions associated
            with the FeatureSelection class.
            
        ValueError
            X_train, X_test, y_train and y_test must have the same shape to use the 
            functions associated with the FeatureSelection class.

        Returns
        -------
        None.

        """
        if X_train.shape[0] == y_train.shape[0] and X_test.shape[0] == y_test.shape[0]:
            
            if isinstance(X_train, np.ndarray) or isinstance(X_train, pd.core.frame.DataFrame):
                self.X_train = X_train
            else:
                raise TypeError(
                    f"'X_train' parameter must be a ndarray or dataframe: got {type(X_train)}"
                    )

            if isinstance(X_test, np.ndarray) or isinstance(X_train, pd.core.frame.DataFrame):
                self.X_test = X_test
            else:
                raise TypeError(
                    f"'X_test' parameter must be a ndarray or dataframe: got {type(X_test)}"
                    )
            
            if isinstance(y_train, pd.core.series.Series) or isinstance(y_train, np.ndarray)\
            or isinstance(y_train, pd.core.frame.DataFrame):
                self.y_train = y_train
            else:
                raise TypeError(
                    f"'y_train' parameter must be a ndarray or series: got {type(y_train)}"
                    )
            
            if isinstance(y_test, pd.core.series.Series) or isinstance(y_test, np.ndarray)\
            or isinstance(y_test, pd.core.frame.DataFrame):
                self.y_test = y_test
            else:
                raise TypeError(
                    f"'y_test' parameter must be a ndarray or series: got {type(y_test)}"
                    )
        
        else:
            raise ValueError(
                "'X_train', 'X_test', 'y_train' and 'y_test,' parameters must have the same \
                shape"
                )

        if isinstance(scoring, str):
            self.scoring = scoring
        else:
            raise TypeError(
                f"'scoring' parameter must be a str: got {type(scoring)}"
                )
        
        if isinstance(n_iter, int):
            self.n_iter = n_iter
        else:
            raise TypeError(
                f"'n_iter' parameter must be an int: got {type(n_iter)}"
                )
        
        if min_features is None or isinstance(min_features, int):
            self.min_features = min_features
        else:
            raise TypeError(
                f"'min_features' parameter must be an int or None: got {type(min_features)}"
                )
        
        if y_lag_select is None or isinstance(y_lag_select, list):
            self.y_lag_select = y_lag_select
        else:
            raise TypeError(
                f"'y_lag_select' parameter must be a list or None: got {type(y_lag_select)}"
                )
        
        if isinstance(random_state, int):
            self.random_state = random_state
        else:
            raise TypeError(
                f"'random_state' parameter must be an int: got {type(random_state)}"
                )
    
        self.df_metrics = pd.DataFrame(
            columns=[
                "Model",
                "MAE",
                "MSE",
                "RMSE",
                "UM",
                "US",
                "UC",
                "U1",
                "U"
            ]
        )
        
        self.reg = LinearRegression(
            fit_intercept=True,
            normalize="deprecated",
            copy_X=True,
            n_jobs=-1,
            positive=False
        )
    
    
    def fit(self):
        """
        Function to fit all sub-models in and out-of-sample.

        Raises
        ------
        ValueError
            if y_lag_select parameter is specified, the list must not be empty.

        Returns
        -------
        sklearn object
            Instance of fitted estimator.

        """
        np.random.seed(self.random_state)
        
        for i in range(self.n_iter):
            list_features_picked = []
            
            if self.y_lag_select is None:
                if self.min_features is None:
                    while len(list_features_picked) <= randint(1, self.X_train.shape[1]):
                        list_features_picked.append(choice(list(self.X_train.columns)))
                else:
                    while len(list_features_picked) <= randint(self.min_features, self.X_train.shape[1]):
                        list_features_picked.append(choice(list(self.X_train.columns)))
            else:
                if len(self.y_lag_select) == 0:
                    raise ValueError(
                        "'y_lag_select' parameter is specified but the list is empty"
                    )
                else:
                    for i in range(0, len(self.y_lag_select)):
                        list_features_picked.append(self.y_lag_select[i])
                
                if self.min_features is None:
                    while len(list_features_picked) <= randint(1, self.X_train.shape[1]):
                        list_features_picked.append(choice(list(self.X_train.columns)))
                else:
                    while len(list_features_picked) <= randint(self.min_features, self.X_train.shape[1]):
                        list_features_picked.append(choice(list(self.X_train.columns)))
            
            list_features_picked = set(list_features_picked)
            list_features_picked = list(list_features_picked)
            
            X_train = self.X_train[list_features_picked]
            X_test = self.X_test[list_features_picked]
            
            self.reg = self.reg.fit(X_train, self.y_train)
            
            self.df_metrics = self.df_metrics.append(
                dict(
                    zip(
                        self.df_metrics.columns,
                        [
                            ", ".join(list_features_picked),
                            Metrics(y_true=self.y_test, y_pred=self.reg.predict(X_test)).mae(),
                            Metrics(y_true=self.y_test, y_pred=self.reg.predict(X_test)).mse(),
                            Metrics(y_true=self.y_test, y_pred=self.reg.predict(X_test)).rmse(),
                            Metrics(y_true=self.y_test, y_pred=self.reg.predict(X_test)).um_theil(),
                            Metrics(y_true=self.y_test, y_pred=self.reg.predict(X_test)).us_theil(),
                            Metrics(y_true=self.y_test, y_pred=self.reg.predict(X_test)).uc_theil(),
                            Metrics(y_true=self.y_test, y_pred=self.reg.predict(X_test)).u1_theil(),
                            Metrics(y_true=self.y_test, y_pred=self.reg.predict(X_test)).u_theil()
                        ]
                    )
                ),
                ignore_index=True
            )
            
        return self.reg
    
    
    def get_params(self) -> dict:
        """
        Function to get sub-model with the best quality of fit (according to a given criterion).

        Returns
        -------
        dict
            Best feature combination.

        """
        check_is_fitted(
            estimator=self.reg,
            msg="Estimator's instance is not fitted yet. Call '.fit()' before using this function"
        )
        
        if self.scoring == "UC":
            self.df_metrics.sort_values(by=self.scoring, ascending=False, inplace=True)
            self.df_metrics.reset_index(drop=True, inplace=True)
            metrics = {
                "Model": self.df_metrics["Model"][0],
                "MAE": self.df_metrics["MAE"][0],
                "MSE": self.df_metrics["MSE"][0],
                "RMSE": self.df_metrics["RMSE"][0],
                "UM": self.df_metrics["UM"][0],
                "US": self.df_metrics["US"][0],
                "UC": self.df_metrics["UC"][0],
                "U1": self.df_metrics["U1"][0],
                "U": self.df_metrics["U"][0]
            }
        else:
            self.df_metrics.sort_values(by=self.scoring, ascending=True, inplace=True)
            self.df_metrics.reset_index(drop=True, inplace=True)
            metrics = {
                "Model": self.df_metrics["Model"][0],
                "MAE": self.df_metrics["MAE"][0],
                "MSE": self.df_metrics["MSE"][0],
                "RMSE": self.df_metrics["RMSE"][0],
                "UM": self.df_metrics["UM"][0],
                "US": self.df_metrics["US"][0],
                "UC": self.df_metrics["UC"][0],
                "U1": self.df_metrics["U1"][0],
                "U": self.df_metrics["U"][0]
            }
        
        return metrics
    
    
    def get_scores(self) -> pd.core.frame.DataFrame:
        """
        Function to get all estimated sub-models and their associated performance metrics.

        Returns
        -------
        pd.core.frame.DataFrame
            Dataframe with all estimated sub-models and their associated metrics.

        """
        check_is_fitted(
            estimator=self.reg,
            msg="Estimator's instance is not fitted yet. Call '.fit()' before using this function"
        )
        
        if self.scoring == "UC":
            self.df_metrics.sort_values(by=self.scoring, ascending=False, inplace=True)
            self.df_metrics.reset_index(drop=True, inplace=True)
        else:
            self.df_metrics.sort_values(by=self.scoring, ascending=True, inplace=True)
            self.df_metrics.reset_index(drop=True, inplace=True)
        
        return self.df_metrics
    
    
    def get_plot(self, date=None, title: str="Estimations in and out-of-sample",
                 xlabel: str="Date", ylabel: str="Value", label1: str="True estimations",
                 color1: str="r", label2: str="Out-sample estimations", color2: str="b",
                 label3: str="In-sample estimations", color3: str="k") -> None:
        """
        Function to display a plot with the true values of the variable of interest and the estimated values 
        on the in and out-of-samples.

        Parameters
        ----------
        date : None or pandas.core.series.Series, optional, default=None
            If date parameter is specified, it will be added as an index for the x-axis of the plot.
            
        title : str, optional, default="Estimations in and out-of-sample"
            Plot's title. Default is "Estimations in and out-of-sample".
            
        xlabel : str, optional, default="Date"
            X-axis title. Default is "Date".
            
        ylabel : str, optional, default="Value"
            Y-axis title. Default is "Value".
            
        label1 : str, optional, default="True estimations"
            Title for truth (correct) target values. Default is "True estimations".
            
        color1 : str, optional, default="r"
            Color for truth (correct) target values. Default is "r" (red).
            
        label2 : str, optional, default="Out-sample estimations"
            Title for out-sample estimated target values. Default is "Out-sample estimations".
            
        color2 : str, optional, default="b"
            Color for out-sample estimated target values. Default is "b" (blue).
            
        label3 : str, optional, default="In-sample estimations"
            Title for in-sample estimated target values. Default is "In-sample estimations".
            
        color3 : str, optional, default="k"
            Color for in-sample estimated target values. Default is "k" (black).

        Raises
        ------
        TypeError
            - date parameter must be a series or None to use get_plot function.
            
            - title parameter must be a str to use get_plot function.
            
            - xlabel parameter must be a str to use get_plot function.
            
            - ylabel parameter must be a str to use get_plot function.
            
            - label1 parameter must be a str to use get_plot function.
            
            - color1 parameter must be a str to use get_plot function.
            
            - label2 parameter must be a str to use get_plot function.
            
            - color2 parameter must be a str to use get_plot function.
            
            - label3 parameter must be a str to use get_plot function.
            
            - color3 parameter must be a str to use get_plot function.

        Returns
        -------
        None.

        """
        if date is None or isinstance(date, pd.core.series.Series):
            pass
        else:
            raise TypeError(
                f"'date' parameter must be a series or None: got {type(date)}"
                )
        
        if isinstance(title, str):
            pass
        else:
            raise TypeError(
                f"'title' parameter must be a str: got {type(title)}"
                )
        
        if isinstance(xlabel, str):
            pass
        else:
            raise TypeError(
                f"'xlabel' parameter must be a str: got {type(xlabel)}"
                )
        
        if isinstance(ylabel, str):
            pass
        else:
            raise TypeError(
                f"'ylabel' parameter must be a str: got {type(ylabel)}"
                )
        
        if isinstance(label1, str):
            pass
        else:
            raise TypeError(
                f"'label1' parameter must be a str: got {type(label1)}"
                )
        
        if isinstance(color1, str):
            pass
        else:
            raise TypeError(
                f"'color1' parameter must be a str: got {type(color1)}"
                )
        
        if isinstance(label2, str):
            pass
        else:
            raise TypeError(
                f"'label2' parameter must be a str: got {type(label2)}"
                )
        
        if isinstance(color2, str):
            pass
        else:
            raise TypeError(
                f"'color2' parameter must be a str: got {type(color2)}"
                )
        
        if isinstance(label3, str):
            pass
        else:
            raise TypeError(
                f"'label3' parameter must be a str: got {type(label3)}"
                )
        
        if isinstance(color3, str):
            pass
        else:
            raise TypeError(
                f"'color3' parameter must be a str: got {type(color3)}"
                )
        
        model = FeatureSelection(
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test,
            scoring=self.scoring,
            n_iter=self.n_iter,
            min_features=self.min_features,
            y_lag_select=self.y_lag_select,
            random_state=self.random_state
            )
        
        model.fit()
        
        params = model.get_params()
        
        list_features_picked = list(params["Model"].split(", "))
        
        X_train = self.X_train[list_features_picked]
        X_test = self.X_test[list_features_picked]
        
        reg = LinearRegression(
            fit_intercept=True,
            normalize="deprecated",
            copy_X=True,
            n_jobs=-1,
            positive=False
        ).fit(
            X_train,
            self.y_train
        )
        
        y_true_values = pd.concat(
            [
                self.y_train.to_frame(name=label1),
                self.y_test.to_frame(name=label1)
            ]
        )
        y_in_sample_values = pd.DataFrame(reg.predict(X_train), columns=[label3])
        y_out_sample_values = pd.concat(
            [
                pd.DataFrame(reg.predict(X_train), columns=[label2]),
                pd.DataFrame(reg.predict(X_test), columns=[label2])
            ],
            axis=0
        )
        
        y_true_values.reset_index(drop=True, inplace=True)
        y_in_sample_values.reset_index(drop=True, inplace=True)
        y_out_sample_values.reset_index(drop=True, inplace=True)
        
        if date is None:
            df_values = pd.concat([y_true_values, y_in_sample_values], axis=1)
            df_values = pd.concat([df_values, y_out_sample_values], axis=1)
            df_values.reset_index(drop=True, inplace=True)
        else:
            df_values = pd.concat([date.to_frame(name="Date"), y_true_values], axis=1)
            df_values = pd.concat([df_values, y_in_sample_values], axis=1)
            df_values = pd.concat([df_values, y_out_sample_values], axis=1)
            df_values.set_index(keys="Date", drop=True, inplace=True)
        
        fig = plt.figure(figsize=(30, 10))
        plt.subplot(1, 1, 1)
        df_values[label1].plot(color=color1, label=label1)
        df_values[label2].plot(color=color2, label=label2)
        df_values[label3].plot(color=color3, label=label3)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.subplots_adjust(hspace=0.3)
        plt.show()
