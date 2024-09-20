import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class EDA:
    def __init__(self, cleaner, train_data=None, test_data=None):
        """
        Initialize the EDA class with a DataCleaner instance.
        :param cleaner: Instance of DataCleaner used to clean the data.
        :param train_data: Initial training dataset (optional).
        :param test_data: Initial testing dataset (optional).
        """
        self.cleaner = cleaner
        self.cleaned_train_data = train_data
        self.cleaned_test_data = test_data

    def clean_data(self):
        """Clean the data using the DataCleaner instance."""
        
        # Ensure that train and test datasets are initialized
        if self.cleaned_train_data is None or self.cleaned_test_data is None:
            raise ValueError("Train and test datasets must be provided before cleaning.")

        # Clean the train and test data
        self.cleaned_train_data, self.cleaned_test_data = self.cleaner.clean_train_and_test(
            self.cleaned_train_data, self.cleaned_test_data
        )
    def check_promo_distribution(self):
        """Check if promos are similarly distributed between train and test sets."""
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
        """Analyze sales before, during, and after holidays."""
        holidays = ['Christmas', 'Easter']  # Add more holidays as necessary
        sales_by_holiday = self.cleaned_train_data.groupby('StateHoliday')['Sales'].mean()

        sns.barplot(x=sales_by_holiday.index, y=sales_by_holiday.values)
        plt.title('Average Sales During Holidays')
        plt.show()

    def find_seasonal_behaviors(self):
        """Find seasonal purchasing behaviors."""
        self.cleaned_train_data['Date'] = pd.to_datetime(self.cleaned_train_data['Date'])
        self.cleaned_train_data['Month'] = self.cleaned_train_data['Date'].dt.month

        sales_by_month = self.cleaned_train_data.groupby('Month')['Sales'].mean()

        plt.figure(figsize=(10, 5))
        sns.lineplot(x=sales_by_month.index, y=sales_by_month.values)
        plt.title("Average Sales by Month (Seasonality)")
        plt.xlabel("Month")
        plt.ylabel("Average Sales")
        plt.show()

    def detect_and_handle_outliers(self):
        """Detect and handle outliers using Z-score or IQR method."""
        z_scores = np.abs(stats.zscore(self.cleaned_train_data.select_dtypes(include=[np.number])))
        outliers = np.where(z_scores > 3)
        print(f"Number of outliers detected: {len(outliers[0])}")
        # Optionally remove outliers based on Z-score threshold
        self.cleaned_train_data = self.cleaned_train_data[(z_scores < 3).all(axis=1)]
        print(f"Data after removing outliers: {self.cleaned_train_data.shape}")

    def handle_missing_data(self):
        """Handle missing values by filling or removing them."""
        missing_data_summary = self.cleaned_train_data.isnull().sum()
        print(f"Missing data summary:\n{missing_data_summary}")

        # Filling missing values (this can be adjusted to the actual logic)
        self.cleaned_train_data.fillna(method='ffill', inplace=True)  # Forward fill as an example
        print("Missing values handled using forward fill.")


    def correlation_sales_customers(self):
        """Calculate and visualize correlation between sales and number of customers."""
        correlation = self.cleaned_train_data[['Sales', 'Customers']].corr()
        sns.heatmap(correlation, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation between Sales and Customers")
        plt.show()

    def promo_effect_on_customers(self):
        """Analyze how promo affects the number of customers."""
        sns.boxplot(x='Promo', y='Customers', data=self.cleaned_train_data)
        plt.title("Promo Effect on Number of Customers")
        plt.show()

    def analyze_store_openings(self):
        """Analyze customer behavior during store opening and closing."""
        sales_by_store_open = self.cleaned_train_data.groupby('Open')['Sales'].mean()

        sns.barplot(x=sales_by_store_open.index, y=sales_by_store_open.values)
        plt.title("Sales during Store Opening/Closing")
        plt.show()

    def assortment_type_sales_analysis(self):
        """Check how the assortment type affects sales."""
        sales_by_assortment = self.cleaned_train_data.groupby('Assortment')['Sales'].mean()

        sns.barplot(x=sales_by_assortment.index, y=sales_by_assortment.values)
        plt.title("Effect of Assortment Type on Sales")
        plt.show()

    def distance_to_competitor_analysis(self):
        """Analyze how the distance to competitors affects sales."""
        sns.scatterplot(x='CompetitionDistance', y='Sales', data=self.cleaned_train_data)
        plt.title("Sales vs Competitor Distance")
        plt.show()

    def generate_summary(self):
        """Summarize key insights from the analysis."""
        print("Key Insights from EDA:")
        print("- Promo distribution is similar/different between training and test sets.")
        print("- Sales tend to increase/decrease during holidays like Christmas and Easter.")
        print("- There is a strong/weak correlation between sales and the number of customers.")
        print("- Promo seems to be attracting more/less new customers.")

    
    def analyze_weekday_vs_weekend_sales(self):
        """Analyze the effect of stores being open on weekdays and weekends."""
        self.cleaned_train_data['Date'] = pd.to_datetime(self.cleaned_train_data['Date'])
        self.cleaned_train_data['DayOfWeek'] = self.cleaned_train_data['Date'].dt.dayofweek
        # Group stores by opening status on weekdays and weekends
        weekday_sales = self.cleaned_train_data[self.cleaned_train_data['DayOfWeek'] < 5].groupby('Store')['Sales'].mean()
        weekend_sales = self.cleaned_train_data[self.cleaned_train_data['DayOfWeek'] >= 5].groupby('Store')['Sales'].mean()

        print("Weekday vs Weekend sales comparison")
        sales_diff = weekday_sales - weekend_sales
        plt.figure(figsize=(10, 6))
        sns.histplot(sales_diff, bins=30, kde=True)
        plt.title("Difference between Weekday and Weekend Sales per Store")
        plt.xlabel("Sales Difference (Weekday - Weekend)")
        plt.ylabel("Number of Stores")
        plt.show()

    def analyze_new_competitors(self):
        """Check how opening/reopening of new competitors affects sales."""
        competitor_na_initial = self.cleaned_train_data[self.cleaned_train_data['CompetitionDistance'].isna()]
        competitor_later = self.cleaned_train_data[~self.cleaned_train_data['CompetitionDistance'].isna()]
        
        print(f"Number of stores with initially missing competitor distance: {len(competitor_na_initial['Store'].unique())}")
        
        # Compare sales for these stores before and after getting competitor data
        sales_before = competitor_na_initial.groupby('Store')['Sales'].mean()
        sales_after = competitor_later.groupby('Store')['Sales'].mean()

        plt.figure(figsize=(10, 6))
        sns.histplot(sales_before - sales_after, bins=30, kde=True)
        plt.title("Sales Difference for Stores Before and After Competitor Opening")
        plt.xlabel("Sales Difference (Before - After)")
        plt.ylabel("Number of Stores")
        plt.show()

    def analyze_effective_promo_deployment(self):
        """Identify stores where promos should be deployed more effectively."""
        # Calculate promo effectiveness by subtracting the average sales without promo from those with promo
        promo_effectiveness = self.cleaned_train_data.groupby('Store').apply(
            lambda x: x[x['Promo'] == 1]['Sales'].mean() - x[x['Promo'] == 0]['Sales'].mean()
        ).reset_index()

        # Rename the columns
        promo_effectiveness.columns = ['Store', 'PromoEffectiveness']

        # Display the top stores where promos are most effective
        print("Top stores where promos are most effective:")
        print(promo_effectiveness.sort_values(by='PromoEffectiveness', ascending=False).head())

        # Plot promo effectiveness for each store
        plt.figure(figsize=(10, 6))
        sns.barplot(x=promo_effectiveness['Store'], y=promo_effectiveness['PromoEffectiveness'])
        plt.title("Promo Effectiveness per Store")
        plt.xlabel("Store")
        plt.ylabel("Sales Increase due to Promo")
        plt.xticks(rotation=90)
        plt.show()


    def run_full_analysis(self):
        """Run all the analysis methods sequentially."""
        self.check_promo_distribution()
        self.analyze_sales_during_holidays()
        self.find_seasonal_behaviors()
        self.correlation_sales_customers()
        self.promo_effect_on_customers()
        self.analyze_store_openings()
        self.assortment_type_sales_analysis()
        self.distance_to_competitor_analysis()
        self.detect_and_handle_outliers()
        self.handle_missing_data()

        self.analyze_weekday_vs_weekend_sales()
        self.analyze_new_competitors()
        self.analyze_effective_promo_deployment()

        self.generate_summary()
