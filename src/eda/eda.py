import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class EDA:
    def __init__(self, cleaner):
        """
        Initialize with a DataCleaner instance to clean training and test datasets.
        :param cleaner: DataCleaner instance responsible for cleaning the data.
        """
        self.cleaner = cleaner
        self.cleaned_train_data = None
        self.cleaned_test_data = None

    def clean_data(self):
            """Clean the data using the DataCleaner instance."""
            self.cleaned_train_data, self.cleaned_test_data = self.cleaner.clean_train_and_test(
                self.cleaner.train_data, self.cleaner.test_data
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
        self.generate_summary()
