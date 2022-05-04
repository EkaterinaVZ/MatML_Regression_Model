import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


class ModelLR:
    def __init__(self,
                 train='Train.csv',
                 test='Test.csv',
                 target='Target.csv',
                 sub='Submission.csv',
                 show_correlation=False):

        self.train = pd.read_csv(train, delimiter=',')
        self.test = pd.read_csv(test, delimiter=',')
        self.target = pd.read_csv(target, delimiter=',')
        self.sub_file = pd.read_csv(sub, delimiter=',')
        self.show_correl = show_correlation
        self.df = pd.concat([self.train, self.test])
        self.cat_columns = []
        self.num_columns = []
        self.lin_reg = LinearRegression(fit_intercept=True)
        self.x_train_, self.y_train_, self.x_val, self.y_val = None, None, None, None
        self.x_test, self.y_test = None, None

    def data_modification(self):
        print('start!!!')
        df_drop = self.df.drop(columns='Unnamed: 0', axis=1)
        self.df = df_drop.drop(columns='period', axis=1)
        if self.show_correl:
            self.show_correlation()
        self.one_hot_coding()

    def show_correlation(self):
        cor = self.df.corr()
        plt.figure(figsize=(10, 7))
        sns.heatmap(cor, annot=True)
        plt.show()

    def one_hot_coding(self):
        self.df = self.df.copy()
        self.df = pd.get_dummies(self.df)
        print(self.df.head(2))
        self.breakdown_data()

    def count_columns(self):
        for column_name in self.df.columns:
            if self.df[column_name].dtypes == object:
                self.cat_columns += [column_name]
            else:
                self.num_columns += [column_name]

    def breakdown_data(self):
        self.count_columns()
        train = self.df.iloc[0:self.train.shape[0], :]
        test = self.df.iloc[self.train.shape[0]:, :]

        x_train = train[self.num_columns].values
        self.x_test = test[self.num_columns].values
        y_train = self.target['polution'].values
        self.x_train_, self.x_val, self.y_train_, self.y_val = \
            train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        self.prediction()

    def prediction(self):
        self.lin_reg.fit(self.x_train_, self.y_train_)
        features_names = self.df.columns
        y_predict = self.lin_reg.predict(self.x_val)
        self.cross_validation()
        self.val_validation(y_predict)
        self.get_regularization(y_predict)

    def cross_validation(self):
        scoring = {'R2': 'r2',
                   '-MSE': 'neg_mean_squared_error',
                   '-MAE': 'neg_mean_absolute_error',
                   'Max': 'max_error'}

        scores = cross_validate(self.lin_reg, self.x_train_, self.y_train_,
                                scoring=scoring, cv=ShuffleSplit(n_splits=5, random_state=42))
        df_cv_linreg = pd.DataFrame(scores)
        print('Результаты Кросс-валидации')
        print('\n')
        print(df_cv_linreg.mean()[2:])
        print('\n')

    def val_validation(self, y_predict):
        self.lin_reg.fit(self.x_train_, self.y_train_)

        print('Ошибка на валидационных данных')
        print('MSE: %.1f' % mse(self.y_val, y_predict))
        print('RMSE: %.1f' % mse(self.y_val, y_predict, squared=False))
        print('R2 : %.4f' % r2_score(self.y_val, y_predict))

    def get_regularization(self, y_predict):
        alpha = 0.01
        model = Ridge(alpha=alpha, max_iter=10000)
        model.fit(self.x_train_, self.y_train_)

        print('Ошибка на валидационных данных после регуляризации')
        print('MSE: %.1f' % mse(self.y_val, y_predict))
        print('RMSE: %.1f' % mse(self.y_val, y_predict, squared=False))
        print('R2 : %.4f' % r2_score(self.y_val, y_predict))
        self.save_pred()

    def save_pred(self):
        # Предсказание на тестовых данных
        self.y_test = self.lin_reg.predict(self.x_test)

        # Записть в столбец 'polution'
        self.sub_file['polution'] = self.y_test
        print('Предсказания на основании тестовых данных:')
        print(self.sub_file['polution'])

        # Перезапись в файл
        self.sub_file.to_csv('My_Submission.csv', index=False)
        print('all')


ModelLR().data_modification()

# parser = argparse.ArgumentParser(description='ModelLReg')
#
# parser.add_argument('--train', type=str, help="name/path (type=str) of the file with train dataset")
# parser.add_argument('--test', type=str, help="name/path (type=str) of the file with test dataset")
# parser.add_argument('--target', type=str, help="name/path (type=str) of the file with target dataset")
# parser.add_argument('--sub', type=str, help="name/path (type=str) of the file with submission dataset")
#
# args = parser.parse_args()
# model = ModelLR(args.train, args.test, args.target, args.sub).data_modification()
# python C:\\Users\\Админ\\PycharmProjects\\Model\\main_model.py --train="Train.csv" --test="Test.csv" --target="Target.csv" --sub="Submission.csv"
# python main_model.py --train="Train.csv" --test="Test.csv" --target="Target.csv" --sub="Submission
