import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class EDA:
    def __init__(self, train_data, test_data, store_data):
        """Initialize with training, test, and store datasets"""
        self.train_data = train_data
        self.test_data = test_data
        self.store_data = store_data
        self.cleaned_train_data = None
        self.cleaned_test_data = None

    def merge_store_data(self):
        """Merge store data with both train and test datasets"""
        self.train_data = pd.merge(self.train_data, self.store_data, on="Store", how="left")
        self.test_data = pd.merge(self.test_data, self.store_data, on="Store", how="left")
        print("Store data merged successfully.")

    def clean_data(self):
        """Clean the training and test datasets (handle missing values, outliers)"""
        self.cleaned_train_data = self._clean_dataset(self.train_data)
        self.cleaned_test_data = self._clean_dataset(self.test_data)

    def _clean_dataset(self, dataset):
        """Internal method to clean individual datasets"""
        # Handle missing values
        dataset['CompetitorDistance'].fillna(dataset['CompetitorDistance'].median(), inplace=True)
        dataset['Promo2SinceWeek'].fillna(0, inplace=True)
        dataset['PromoInterval'].fillna('None', inplace=True)
        
        # Handle outliers using Z-score method
        z_scores = np.abs(stats.zscore(dataset.select_dtypes(include=np.number)))
        dataset_cleaned = dataset[(z_scores < 3).all(axis=1)]
        
        return dataset_cleaned

    def check_promo_distribution(self):
        """Check if promos are similarly distributed between train and test sets"""
        train_promo_distribution = self.cleaned_train_data['Promo'].value_counts(normalize=True)
        test_promo_distribution = self.cleaned_test_data['Promo'].value_counts(normalize=True)
        
        print("Training Promo Distribution:\n", train_promo_distribution)
        print("Test Promo Distribution:\n", test_promo_distribution)

        sns.barplot(x=train_promo_distribution.index, y=train_promo_distribution.values, color="blue", alpha=0.5, label="Train")
        sns.barplot(x=test_promo_distribution.index, y=test_promo_distribution.values, color="orange", alpha=0.5, label="Test")
        plt.title("Promo Distribution in Train and Test Sets")
        plt.legend()
        plt.show()

    def analyze_sales_during_holidays(self):
        """Analyze sales before, during, and after holidays"""
        holidays = ['Christmas', 'Easter']  # Add more holidays as necessary
        sales_by_holiday = self.cleaned_train_data.groupby('Holiday')['Sales'].mean()
        
        sns.barplot(x=sales_by_holiday.index, y=sales_by_holiday.values)
        plt.title('Average Sales During Holidays')
        plt.show()

    def find_seasonal_behaviors(self):
        """Find seasonal purchasing behaviors"""
        self.cleaned_train_data['Date'] = pd.to_datetime(self.cleaned_train_data['Date'])
        self.cleaned_train_data['Month'] = self.cleaned_train_data['Date'].dt.month
        
        sales_by_month = self.cleaned_train_data.groupby('Month')['Sales'].mean()
        
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=sales_by_month.index, y=sales_by_month.values)
        plt.title("Average Sales by Month (Seasonality)")
        plt.xlabel("Month")
        plt.ylabel("Average Sales")
        plt.show()

    def correlation_sales_customers(self):
        """Calculate and visualize correlation between sales and number of customers"""
        correlation = self.cleaned_train_data[['Sales', 'Customers']].corr()
        sns.heatmap(correlation, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation between Sales and Customers")
        plt.show()

    def promo_effect_on_customers(self):
        """Analyze how promo affects the number of customers"""
        sns.boxplot(x='Promo', y='Customers', data=self.cleaned_train_data)
        plt.title("Promo Effect on Number of Customers")
        plt.show()

    def analyze_store_openings(self):
        """Analyze customer behavior during store opening and closing"""
        sales_by_store_open = self.cleaned_train_data.groupby('StoreOpen')['Sales'].mean()

        sns.barplot(x=sales_by_store_open.index, y=sales_by_store_open.values)
        plt.title("Sales during Store Opening/Closing")
        plt.show()

    def assortment_type_sales_analysis(self):
        """Check how the assortment type affects sales"""
        sales_by_assortment = self.cleaned_train_data.groupby('Assortment')['Sales'].mean()

        sns.barplot(x=sales_by_assortment.index, y=sales_by_assortment.values)
        plt.title("Effect of Assortment Type on Sales")
        plt.show()

    def distance_to_competitor_analysis(self):
        """Analyze how the distance to competitors affects sales"""
        sns.scatterplot(x='CompetitorDistance', y='Sales', data=self.cleaned_train_data)
        plt.title("Sales vs Competitor Distance")
        plt.show()

    def generate_summary(self):
        """Summarize key insights from the analysis"""
        print("Key Insights from EDA:")
        print("- Promo distribution is similar/different between training and test sets.")
        print("- Sales tend to increase/decrease during holidays like Christmas and Easter.")
        print("- There is a strong/weak correlation between sales and the number of customers.")
        print("- Promo seems to be attracting more/less new customers.")

    def run_full_analysis(self):
        """Run all the analysis methods sequentially"""
        # self.merge_store_data()
        # self.clean_data()
        self.check_promo_distribution()
        self.analyze_sales_during_holidays()
        self.find_seasonal_behaviors()
        self.correlation_sales_customers()
        self.promo_effect_on_customers()
        self.analyze_store_openings()
        self.assortment_type_sales_analysis()
        self.distance_to_competitor_analysis()
        self.generate_summary()

# # Example Usage
# if __name__ == "__main__":
#     # Load datasets
#     train_data = pd.read_csv('train.csv')
#     test_data = pd.read_csv('test.csv')
#     store_data = pd.read_csv('store.csv')
    
#     # Perform customer behavior analysis
#     analysis = CustomerBehaviorAnalysis(train_data, test_data, store_data)
#     analysis.run_full_analysis()
