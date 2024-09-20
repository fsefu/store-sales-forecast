import pandas as pd

class DataCleaner:
    def __init__(self, store_data):
        """
        Initialize the DataCleaner with store data for merging.
        :param store_data: DataFrame containing store-related information.
        """
        self.store_data = store_data

    def get_columns(self, dataset):
        """
        Get and print all columns in the dataset before cleaning.
        :param dataset: DataFrame to inspect.
        :return: List of column names.
        """
        columns = list(dataset.columns)
        print("Columns before cleaning:", columns)
        return columns

    def clean_dataset(self, dataset):
        """
        Clean the dataset by handling missing values and outliers.
        :param dataset: DataFrame to be cleaned.
        :return: Cleaned DataFrame.
        """

        # Ensure 'Date' column is in the correct datetime format
        if 'Date' in dataset.columns:
            dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')

        # Fill missing or invalid values in relevant columns
        fill_values = {
            'Sales': 0,  # Sales: Fill missing values with 0
            'Customers': dataset['Customers'].median() if 'Customers' in dataset.columns else None,
            'Open': 1,  # Assume missing 'Open' means store is open
            'Promo': 0,  # Assume missing 'Promo' means no promo
            'StateHoliday': '0',  # Replace missing state holidays with '0' (no holiday)
            'SchoolHoliday': 0,  # Fill missing school holidays with 0
            'StoreType': dataset['StoreType'].mode()[0] if 'StoreType' in dataset.columns else None,
            'Assortment': dataset['Assortment'].mode()[0] if 'Assortment' in dataset.columns else None,
            'CompetitionDistance': dataset['CompetitionDistance'].median() if 'CompetitionDistance' in dataset.columns else None,
            'CompetitionOpenSinceMonth': 0,  # Fill missing competition month with 0
            'CompetitionOpenSinceYear': 0,  # Fill missing competition year with 0
            'Promo2': 0,  # Assume missing 'Promo2' means no participation
            'Promo2SinceWeek': 0,  # Fill missing 'Promo2SinceWeek' with 0
            'Promo2SinceYear': 0,  # Fill missing 'Promo2SinceYear' with 0
            'PromoInterval': 'None'  # Fill missing 'PromoInterval' with 'None'
        }

        for column, value in fill_values.items():
            if column in dataset.columns:
                dataset[column].fillna(value, inplace=True)

        # Ensure numeric columns are in the correct format
        numeric_columns = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']
        for column in numeric_columns:
            if column in dataset.columns:
                dataset[column] = pd.to_numeric(dataset[column], errors='coerce')

        return dataset

    def merge_store_data(self, train_data, test_data):
        """
        Merge store data with the train and test datasets.
        :param train_data: Training DataFrame.
        :param test_data: Testing DataFrame.
        :return: Merged training and testing DataFrames.
        """
        merged_train = pd.merge(train_data, self.store_data, on="Store", how="left")
        merged_test = pd.merge(test_data, self.store_data, on="Store", how="left")
        return merged_train, merged_test

    def clean_train_and_test(self, train_data, test_data):
        """
        Merge and clean both training and test datasets.
        :param train_data: Training DataFrame.
        :param test_data: Testing DataFrame.
        :return: Cleaned training and testing DataFrames.
        """
        # Step 1: Merge with store data
        merged_train_data, merged_test_data = self.merge_store_data(train_data, test_data)

        # Step 2: Clean the merged datasets
        cleaned_train_data = self.clean_dataset(merged_train_data)
        cleaned_test_data = self.clean_dataset(merged_test_data)

        return cleaned_train_data, cleaned_test_data

