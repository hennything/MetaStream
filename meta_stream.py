from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

import argparse
import pathlib
import pandas as pd
import numpy as np
import scipy

from util import nmse, percentage_difference

TIE = 'tie'

class MetaStream():

    # TODO: add error message if the size of the initial meta-table has less instances than features
    def __init__(self, meta_learner, learners, base_window=100, base_sel_window_size=10, meta_window=200, strategy=None, threshold=None):

        self.meta_learner = meta_learner
        self.learners = learners

        self.base_window = base_window
        self.base_sel_window_size = base_sel_window_size
        self.meta_window = meta_window
        self.strategy = strategy
        self.threshold = threshold
        self.num_learners = len(self.learners)

        if self.strategy != None and len(self.learners) > 2:
            raise ValueError("When using 'tie' or 'combination' strategy MetaStream must receive exactly 2 learners")
        elif self.strategy != None and self.strategy != TIE and self.strategy != 'combination':
            raise ValueError("strategy can only be 1 of 3 options (None, 'tie', 'combination')")
        elif self.strategy == TIE and self.threshold == None:
            raise ValueError("When using 'tie' strategy threshold must be provided")

        self.meta_table = pd.DataFrame()

        # # NOTE: keeps track of current location in base_data
        # self.base_index = 0
        # self.meta_index = 0


    # TODO: allow user input for which meta-features to inspect
    def meta_features(self, X_train, y_train, X_sel):

        temp = {}

        # train window 
        for i, col in enumerate(X_train):
            temp.update({"X_train_mean_"+str(i): np.mean(X_train[col])})
            temp.update({"X_train_var_"+str(i): np.var(X_train[col])})
            temp.update({"X_train_min_"+str(i): np.min(X_train[col])})
            temp.update({"X_train_max_"+str(i): np.max(X_train[col])})
            temp.update({"X_train_median_"+str(i): np.median(X_train[col])})

        # train window
        temp.update({"y_train_mean_"+str(i): np.mean(y_train)})
        temp.update({"y_train_var_"+str(i): np.var(y_train)})
        temp.update({"y_train_min_"+str(i): np.min(y_train)})
        temp.update({"y_train_max_"+str(i): np.max(y_train)})
        temp.update({"y_train_median_"+str(i): np.median(y_train)})

        # selection window
        for i, col in enumerate(X_sel):
            temp.update({"X_sel_mean_"+str(i): np.mean(X_sel[col])})
            temp.update({"X_sel_var_"+str(i): np.var(X_sel[col])})
            temp.update({"X_sel_min_"+str(i): np.min(X_sel[col])})
            temp.update({"X_sel_max_"+str(i): np.max(X_sel[col])})
            temp.update({"X_sel_median_"+str(i): np.median(X_sel[col])})

        # selection window
        # correlation between numerical attributes
        for i, col_i in enumerate(X_sel):
            for j, col_j in enumerate(X_sel):
                if i != j:
                    temp.update({"X_sel_corr_"+str(i)+"_"+str(j) : np.correlate(X_sel[col_i], X_sel[col_j])})

        # train windown
        # correlation between numberical attributes and target
        for i, col in enumerate(X_train):
            temp.update({"X_train_corr_"+str(i) : np.correlate(X_train[col], y_train)})

        # skewness and kurtosis of numeric attrributes
        for i, col in enumerate(X_sel):
            temp.update({"X_sel_skew_"+str(i): scipy.stats.skew(X_sel[col])})
            temp.update({"X_sel_kurtosis_"+str(i): scipy.stats.kurtosis(X_sel[col])})


        # print(temp)
        # print(len(temp))
        return temp
             

        # temp = {}
        
        # with localconverter(ro.default_converter + pandas2ri.converter):
            # ecol = importr("ECoL")
            # rfeatures = ecol.complexity(X, y, summary=["mean"])
            
            # for i, value in enumerate(rfeatures):
                # temp.update({str(i) : value})

        # return temp

    def ensemble(self, X_train, y_train, X_sel, y_sel):
        scores = [learner.fit(X_train, y_train).predict(X_sel) for learner in self.learners]
        scores = np.array([sum(i)/len(self.learners) for i in zip(*scores)])
        score = nmse(list(y_sel), scores)
        return score

   # NOTE: initial base fit   
    def base_train(self, data, target):
        """
        params:
        - data: pandas dataframe
        - target: string
        """
        for idx in range(self.meta_window):

            train = data.iloc[idx * self.base_sel_window_size : idx * self.base_sel_window_size + self.base_window]
            sel = data.iloc[idx * self.base_sel_window_size + self.base_window : (idx + 1) * self.base_sel_window_size + self.base_window]

            X_train, y_train = train.drop(target, axis=1), train[target]
            X_sel, y_sel = sel.drop(target, axis=1), sel[target]

            meta_features = self.meta_features(X_train, y_train, X_sel)
            self.base_fit(X_train, y_train)

            preds = self.base_predict(X_sel)
            scores = [nmse(pred, y_sel) for pred in preds]
            if self.strategy == None:
                meta_features.update({'regressor' : np.argmin(scores)})    
            elif self.strategy == TIE:
                if percentage_difference(scores) <= self.threshold:
                    meta_features.update({'regressor' : self.num_learners})
                else:
                    meta_features.update({'regressor' : np.argmin(scores)})

            self.meta_table = self.meta_table.append(meta_features, ignore_index=True)

        # self.base_index = idx
        print(self.meta_table)

    # TODO: make private
    # NOTE: base fit is performed on the base-learners
    def base_fit(self, X, y):
        [learner.fit(X, y) for learner in self.learners]
    
    # TODO: make private
    # NOTE: base predict is performed on the base-learners
    def base_predict(self, X):
        """
        returns: a prediction for each of the base-learners
        """
        return [learner.predict(X) for learner in self.learners]


    # TODO: rename function name
    def meta_train(self, data, target, default=False, ensemble=False):

        # need to do initial meta_fit
        self.meta_fit(self.meta_table.drop(['regressor'], axis=1), self.meta_table['regressor'])
        
        max_data_size = int((data.shape[0] - self.base_window) / self.base_sel_window_size)
        print(max_data_size)

        if default: 
            default_scores = []
            default_recommended = []
        
        if ensemble: ensemble_scores = []
        # 
        m_recommended = []
        score_recommended = []
        m_actual = []

        for idx in range(self.meta_window, max_data_size):
            
            # print(idx)

            train = data.iloc[idx * self.base_sel_window_size : idx * self.base_sel_window_size + self.base_window]
            sel = data.iloc[idx * self.base_sel_window_size + self.base_window : (idx + 1) * self.base_sel_window_size + self.base_window]

            X_train, y_train = train.drop(target, axis=1), train[target]
            X_sel, y_sel = sel.drop(target, axis=1), sel[target]

            meta_features = self.meta_features(X_train, y_train, X_sel)
            pred = int(self.meta_predict(np.array(list(meta_features.values()), dtype=object).reshape(1, -1)))
            m_recommended.append(pred)


            if self.strategy == None: 
                score = nmse(list(y_sel), self.learners[pred].fit(X_train, y_train).predict(X_sel))
                score_recommended.append(score)
            
            elif self.strategy == TIE:
                score_recommended.append(self.ensemble(X_train, y_train, X_sel, y_sel))

            # TODO: generate output for default method
            if default:
                default_learner = int(self.meta_table['regressor'].value_counts().idxmax())
                default_recommended.append(default_learner)
                if default_learner == self.num_learners and self.strategy != None:
                    default_scores.append(self.ensemble(X_train, y_train, X_sel, y_sel))
                else:
                    default_score = nmse(list(y_sel), self.learners[default_learner].fit(X_train, y_train).predict(X_sel))
                    default_scores.append(default_score)

            # TODO: make seperate function for this method
            if ensemble:
                ensemble_scores.append(self.ensemble(X_train, y_train, X_sel, y_sel))

            self.base_fit(X_train, y_train)
            preds = self.base_predict(X_sel)
            scores = [nmse(pred, y_sel) for pred in preds]
            if self.strategy == None:
                regressor = np.argmin(scores)
                m_actual.append(np.argmin(regressor))
                meta_features.update({'regressor' : np.argmin(regressor)})

            elif self.strategy == 'tie':
                if percentage_difference(scores) > self.threshold:
                    regressor = np.argmin(scores)
                    m_actual.append(regressor)
                    meta_features.update({'regressor' : regressor})
                    # print('regressor: ', regressor, scores)
                else:
                    m_actual.append(2)
                    meta_features.update({'regressor' : self.num_learners})
                    # print('tie: ', scores)
            
            # sliding window for the meta-table
            self.meta_fit(self.meta_table.drop(['regressor'], axis=1)[-self.meta_window:], self.meta_table['regressor'][-self.meta_window:])

        print("Mean score Recommended {:.2f}+-{:.2f}".format(np.mean(score_recommended), np.std(score_recommended)))
        print(len([i for i, j in zip(m_actual, m_recommended) if i == j]) / len(m_actual))
        
        if default:
            print("Mean score default {:.2f}+-{:.2f}".format(np.mean(default_scores), np.std(default_scores)))
            # print(len(m_actual), len(default_recommended))
            # print(default_recommended)
            print(len([i for i, j in zip(m_actual, default_recommended) if i == j]) / len(m_actual))

        if ensemble:
            print("Mean score ensemble {:.2f}+-{:.2f}".format(np.mean(ensemble_scores), np.std(ensemble_scores)))


    # TODO: make private
    # NOTE: meta fit is performed on the meta-learner
    def meta_fit(self, X, y):
        self.meta_learner.fit(X, y)

    # TODO: make private
    # NOTE: meta predict is performed on the meta-learner
    def meta_predict(self, X):
        """
        returns: a prediction for the meta-learner
        """
        return self.meta_learner.predict(X)


# TODO: need to remove this after experiimentation
if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', type=pathlib.Path)

    parser.add_argument('-base_data_window', default=100, type=int)
    parser.add_argument('-base_sel_window_size', default=10, type=int)
    parser.add_argument('-meta_data_window', default=200, type=int)

    base_data_window = parser.parse_args().base_data_window
    base_sel_window_size = parser.parse_args().base_sel_window_size
    meta_data_window = parser.parse_args().meta_data_window

    # print(base_data_window)
    # print(base_sel_window_size)
    # print(meta_data_window)

    df = pd.read_csv(parser.parse_args().datapath)
    df = df[['period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']]
    # df['class'] = (df['class'] == "UP").astype(int)


    # NOTE: list of regression algorithms
    models =    [
                # SVR(),
                RandomForestRegressor(random_state=42),
                # LinearRegression(),
                # Lasso(),
                # Ridge(),
                GradientBoostingRegressor(random_state=42)
                ]


    # NOTE: meta-learner
    meta_learner = SGDClassifier()

    metas = MetaStream(meta_learner, models, base_data_window, base_sel_window_size, meta_data_window, strategy='tie', threshold=.05)

    # creates baseline meta-data
    metas.base_train(data=df, target='nswdemand')
    # print(metas.meta_table)
    metas.meta_train(data=df, target='nswdemand', default=True, ensemble=True)


