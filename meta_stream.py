from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import argparse
import pathlib
import pandas as pd
import numpy as np
import scipy

from util import nmse, percentage_difference

TIE = 'tie'
COMBINATION = 'combination'

class MetaStream():

    # TODO: add error message if the size of the initial meta-table has less instances than features
    def __init__(self, meta_learner, learners, base_window=100, base_delay_window=0, base_sel_window=10, meta_window=200, strategy=None, default=False, ensemble=False, threshold=None, pairs=False):

        self.meta_learner = meta_learner
        self.learners = learners

        self.base_window = base_window
        self.base_delay_window = base_delay_window
        self.base_sel_window = base_sel_window
        self.meta_window = meta_window
        self.strategy = strategy
        self.default = default
        if self.default: 
            self.default_scores = []
            self.default_recommended = []

        self.ensemble = ensemble
        if self.ensemble: self.ensemble_scores = []
        self.threshold = threshold
        self.pairs = pairs
        if self.pairs:
            self.reg_1_scores = []
            self.reg_2_scores = []
        self.num_learners = len(self.learners)

        if self.strategy != COMBINATION and len(self.learners) > 2:
            raise ValueError("when using 'tie' strategy MetaStream must receive exactly 2 learners")
        elif self.strategy != None and self.strategy != TIE and self.strategy != COMBINATION:
            raise ValueError("strategy can only be 1 of 3 options (None, 'tie', 'combination')")
        elif self.strategy == TIE and self.threshold == None:
            raise ValueError("When using 'tie' strategy threshold must be provided")
        elif self.pairs == True and self.num_learners > 2:
            raise ValueError("full regressor report is only generated for pairs of regressors")

        self.meta_table = pd.DataFrame()

        self.recommended = []
        self.score_recommended = []
        self.actual = []

    def _meta_features(self, X_train, y_train, X_sel):

        temp = {}

        # train window 
        for i, col in enumerate(X_train):
            X_train_percentiles = np.percentile(X_train[col], (25, 75))
            X_train_iqr = X_train_percentiles[1] - X_train_percentiles[0]

            temp.update({"X_train_mean_"+str(i): np.mean(X_train[col])})
            temp.update({"X_train_var_"+str(i): np.var(X_train[col])})
            temp.update({"X_train_min_"+str(i): np.min(X_train[col])})
            temp.update({"X_train_max_"+str(i): np.max(X_train[col])})
            temp.update({"X_train_median_"+str(i): np.median(X_train[col])})
            temp.update({"X_sel_dispersion_"+str(i): X_train_iqr})

        # train window
        y_train_percentiles = np.percentile(y_train, (25, 75))
        y_train_iqr = y_train_percentiles[1] - y_train_percentiles[0]
        y_train_lower_outliers = np.where(y_train < y_train_percentiles[0] - 1.5 * y_train_iqr)[0]
        y_train_upper_outliers = np.where(y_train > y_train_percentiles[1] + 1.5 * y_train_iqr)[0]
        y_train_prob_outliers = (len(y_train_lower_outliers) + len(y_train_upper_outliers)) / len(y_train)

        temp.update({"y_train_mean_"+str(i): np.mean(y_train)})
        temp.update({"y_train_var_"+str(i): np.var(y_train)})
        temp.update({"y_train_min_"+str(i): np.min(y_train)})
        temp.update({"y_train_max_"+str(i): np.max(y_train)})
        temp.update({"y_train_median_"+str(i): np.median(y_train)})
        temp.update({"y_train_prob_outliers_"+str(i): y_train_prob_outliers})
        temp.update({"y_train_dispersion_"+str(i): y_train_iqr})

        # selection window
        for i, col in enumerate(X_sel):
            X_sel_percentiles = np.percentile(X_sel[col], (25, 75))
            X_sel_iqr = X_sel_percentiles[1] - X_sel_percentiles[0]
            X_sel_lower_outliers = np.where(X_sel[col] < X_sel_percentiles[0] - 1.5 * X_sel_iqr)[0]
            X_sel_upper_outliers = np.where(X_sel[col] > X_sel_percentiles[1] + 1.5 * X_sel_iqr)[0]
            X_sel_prob_outliers = (len(X_sel_lower_outliers) + len(X_sel_upper_outliers)) / len(X_sel)

            temp.update({"X_sel_mean_"+str(i): np.mean(X_sel[col])})
            temp.update({"X_sel_var_"+str(i): np.var(X_sel[col])})
            temp.update({"X_sel_min_"+str(i): np.min(X_sel[col])})
            temp.update({"X_sel_max_"+str(i): np.max(X_sel[col])})
            temp.update({"X_sel_median_"+str(i): np.median(X_sel[col])})
            temp.update({"X_sel_prob_outliers_"+str(i): X_sel_prob_outliers})

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

        return temp

    def _ensemble(self, X_train, y_train, X_sel, y_sel):
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

            train = data.iloc[idx * self.base_sel_window : idx * self.base_sel_window + self.base_window]
            sel = data.iloc[self.base_delay_window + idx * self.base_sel_window + self.base_window : (idx + 1) * self.base_sel_window + self.base_window]
            # print(train.index)
            # print(sel.index)
            X_train, y_train = train.drop(target, axis=1), train[target]
            X_sel, y_sel = sel.drop(target, axis=1), sel[target]

            meta_features = self._meta_features(X_train, y_train, X_sel)
            self._base_fit(X_train, y_train)

            preds = self._base_predict(X_sel)
            scores = [nmse(pred, y_sel) for pred in preds]
            if self.strategy == None:
                meta_features.update({'regressor' : np.argmin(scores)})    
            elif self.strategy == TIE:
                if percentage_difference(scores) <= self.threshold:
                    meta_features.update({'regressor' : self.num_learners})
                else:
                    meta_features.update({'regressor' : np.argmin(scores)})
            elif self.strategy == COMBINATION:
                pred_combination = np.array([sum(i)/len(self.learners) for i in zip(*preds)])
                scores.append(nmse(pred_combination, y_sel))
                meta_features.update({'regressor' : np.argmin(scores)})

            self.meta_table = self.meta_table.append(meta_features, ignore_index=True)

    # NOTE: base fit is performed on the base-learners
    def _base_fit(self, X, y):
        """
        fit the base-learners using training data
        """
        [learner.fit(X, y) for learner in self.learners]
    
    # NOTE: base predict is performed on the base-learners
    def _base_predict(self, X):
        """
        returns: a prediction for each of the base-learners
        """
        return [learner.predict(X) for learner in self.learners]

    # TODO: could probably optimize by only running each regressor once uniquely
    def meta_train(self, data, target):

        # initial meta-fit
        self._meta_fit(self.meta_table.drop(['regressor'], axis=1), self.meta_table['regressor'])
        
        max_data_size = int((data.shape[0] - self.base_window) / self.base_sel_window)

        for idx in range(self.meta_window, max_data_size):

            train = data.iloc[idx * self.base_sel_window : idx * self.base_sel_window + self.base_window]
            sel = data.iloc[self.base_delay_window + idx * self.base_sel_window + self.base_window : (idx + 1) * self.base_sel_window + self.base_window]

            X_train, y_train = train.drop(target, axis=1), train[target]
            X_sel, y_sel = sel.drop(target, axis=1), sel[target]

            meta_features = self._meta_features(X_train, y_train, X_sel)
            pred = int(self._meta_predict(np.array(list(meta_features.values()), dtype=object).reshape(1, -1)))
            self.recommended.append(pred)

            if self.strategy != None and pred == self.num_learners:
               self.score_recommended.append(self._ensemble(X_train, y_train, X_sel, y_sel))
            else:
                score = nmse(list(y_sel), self.learners[pred].fit(X_train, y_train).predict(X_sel))
                self.score_recommended.append(score)

            if self.pairs:
                scores = [learner.fit(X_train, y_train).predict(X_sel) for learner in self.learners]
                self.reg_1_scores.append(nmse(list(y_sel), scores[0]))
                self.reg_2_scores.append(nmse(list(y_sel), scores[1]))

            if self.default:
                default_learner = int(self.meta_table['regressor'].value_counts().idxmax())
                self.default_recommended.append(default_learner)
                if default_learner == self.num_learners and self.strategy != None:
                    self.default_scores.append(self._ensemble(X_train, y_train, X_sel, y_sel))
                else:
                    default_score = nmse(list(y_sel), self.learners[default_learner].fit(X_train, y_train).predict(X_sel))
                    self.default_scores.append(default_score)

            if self.ensemble:
                self.ensemble_scores.append(self._ensemble(X_train, y_train, X_sel, y_sel))

            self._base_fit(X_train, y_train)
            preds = self._base_predict(X_sel)
            scores = [nmse(pred, y_sel) for pred in preds]
            if self.strategy == None:
                regressor = np.argmin(scores)
                self.actual.append(np.argmin(regressor))
                meta_features.update({'regressor' : np.argmin(regressor)})

            elif self.strategy == 'tie':
                if percentage_difference(scores) > self.threshold:
                    regressor = np.argmin(scores)
                    self.actual.append(regressor)
                    meta_features.update({'regressor' : regressor})
                else:
                    self.actual.append(self.num_learners)
                    meta_features.update({'regressor' : self.num_learners})

            elif self.strategy == 'combination':
                pred_combination = np.array([sum(i)/len(self.learners) for i in zip(*preds)])
                scores.append(nmse(pred_combination, y_sel))
                regressor = np.argmin(scores)
                self.actual.append(regressor)
                meta_features.update({'regressor' : regressor})

            # sliding window for the meta-table
            self._meta_fit(self.meta_table.drop(['regressor'], axis=1)[-self.meta_window:], self.meta_table['regressor'][-self.meta_window:])

    # NOTE: meta fit is performed on the meta-learner
    def _meta_fit(self, X, y):
        """
        fit the meta-learner using training data
        """
        self.meta_learner.fit(X.values, y.values)

    # NOTE: meta predict is performed on the meta-learner
    def _meta_predict(self, X):
        """
        returns: a prediction for the meta-learner
        """
        return self.meta_learner.predict(X)

    def get_results(self):
        base_results = {}
        meta_results = {}
        
        base_results['recommended'] = np.mean(self.score_recommended)
        meta_results['recommended'] = 1 - len([i for i, j in zip(self.actual, self.recommended) if i == j]) / len(self.actual)

        if self.default:
            base_results['default'] = np.mean(self.default_scores)
            meta_results['default'] = 1 - len([i for i, j in zip(self.actual, self.default_recommended) if i == j]) / len(self.actual)

        if self.ensemble:
            base_results['ensemble'] = np.mean(self.ensemble_scores)

        if self.pairs:
            base_results['regressor 1'] = np.mean(self.reg_1_scores)
            base_results['regressor 2'] = np.mean(self.reg_2_scores)

        return base_results, meta_results

    def print_results(self):

        print("Mean score recommended {:.3f}+-{:.3f}".format(np.mean(self.score_recommended), np.std(self.score_recommended)))
        print("Meta-level score recommended {:.3f}".format(1 - len([i for i, j in zip(self.actual,self.recommended) if i == j]) / len(self.actual)))

        if self.default:
            print("Mean score default {:.3f}+-{:.3f}".format(np.mean(self.default_scores), np.std(self.default_scores)))
            print("Meta-level score default {:.3f}".format(1 - len([i for i, j in zip(self.actual, self.default_recommended) if i == j]) / len(self.actual)))

        if self.ensemble:
            print("Mean score ensemble {:.3f}+-{:.3f}".format(np.mean(self.ensemble_scores), np.std(self.ensemble_scores)))

        if self.pairs:
            print("Mean score regressor 1 {:.3f}".format(np.mean(self.reg_1_scores)))
            print("Mean score regressor 2 {:.3f}".format(np.mean(self.reg_2_scores)))


# TODO: need to remove this after experiimentation
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', type=pathlib.Path)
    
    parser.add_argument('-base_data_window', default=100, type=int)
    parser.add_argument('-base_delay_window', default=0, type=int)
    parser.add_argument('-base_sel_window', default=10, type=int)
    parser.add_argument('-meta_data_window', default=200, type=int)

    base_data_window = parser.parse_args().base_data_window
    base_delay_window = parser.parse_args().base_delay_window
    base_sel_window = parser.parse_args().base_sel_window
    meta_data_window = parser.parse_args().meta_data_window

    df = pd.read_csv(parser.parse_args().datapath)
    df = df[['period', 'nswprice', 'nswdemand', 'vicprice', 'vicdemand', 'transfer']]

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
    meta_learner = RandomForestClassifier()

    metas = MetaStream(meta_learner, models, base_data_window, base_delay_window, base_sel_window, meta_data_window, strategy='combination', default=True, ensemble=True, pairs=True, threshold=0.1)

    # creates baseline meta-data
    metas.base_train(data=df, target='nswdemand')

    metas.meta_train(data=df, target='nswdemand')

    metas.print_results()

    base, meta = metas.get_results()
    print(base)
    print(meta)




