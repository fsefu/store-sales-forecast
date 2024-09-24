import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.
        """
        self.df = df
        self.scaler = StandardScaler()

    def fill_missing_values(self):
        """
        Fill missing values with appropriate strategies.
        """
        self.df['CompetitionDistance'].fillna(self.df['CompetitionDistance'].median(), inplace=True)
        self.df['Promo2SinceYear'].fillna(0, inplace=True)
        self.df['Promo2SinceWeek'].fillna(0, inplace=True)
        self.df['CompetitionOpenSinceYear'].fillna(self.df['CompetitionOpenSinceYear'].mode()[0], inplace=True)
        self.df['CompetitionOpenSinceMonth'].fillna(self.df['CompetitionOpenSinceMonth'].mode()[0], inplace=True)
        self.df['PromoInterval'].fillna('None', inplace=True)

    def convert_to_numeric(self):
        """
        Convert categorical variables to numeric using one-hot encoding.
        """
        self.df = pd.get_dummies(self.df, columns=['StoreType', 'Assortment', 'StateHoliday'], drop_first=True)

    def extract_datetime_features(self):
        """
        Extract features from datetime columns.
        """
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        self.df['Weekend'] = self.df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        self.df['Month'] = self.df['Date'].dt.month
        self.df['DayOfMonth'] = self.df['Date'].dt.day
        self.df['IsMonthStart'] = self.df['Date'].dt.is_month_start.astype(int)
        self.df['IsMonthEnd'] = self.df['Date'].dt.is_month_end.astype(int)
        # Example of days to holidays or special dates - adjust as needed
        self.df['DaysToHoliday'] = (pd.to_datetime('2024-12-25') - self.df['Date']).dt.days

    def merge_store_data(self, store_df):
        """
        Merge store data into the main DataFrame.
        """
        self.df = self.df.merge(store_df, how='left', on='Store')

    def scale_features(self):
        """
        Scale the numeric features using StandardScaler.
        """
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_columns] = self.scaler.fit_transform(self.df[numeric_columns])

    def preprocess(self, store_df=None):
        """
        Execute the full preprocessing pipeline. Optionally merge store data.
        """
        self.fill_missing_values()
        self.convert_to_numeric()
        self.extract_datetime_features()
        if store_df is not None:
            self.merge_store_data(store_df)
        self.scale_features()

        return self.df

# # Example usage inside the notebook:
# if __name__ == "__main__":
#     # Load the datasets
#     train_data = pd.read_csv('../data/train.csv')
#     test_data = pd.read_csv('../data/test.csv')
#     store_data = pd.read_csv('../data/store.csv')

#     # Preprocess the train data
#     preprocessor_train = DataPreprocessor(train_data)
#     processed_train_data = preprocessor_train.preprocess(store_df=store_data)
    
#     # Preprocess the test data
#     preprocessor_test = DataPreprocessor(test_data)
#     processed_test_data = preprocessor_test.preprocess(store_df=store_data)

#     # Print a preview of the processed data
#     print(processed_train_data.head())
#     print(processed_test_data.head())
