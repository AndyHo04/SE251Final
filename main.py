import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from numpy.polynomial.polynomial import Polynomial


class Pipeline:
    def __init__(self, data: pd.DataFrame):
        self.data: pd.DataFrame = data
        self.headers = {}

    def set_headers(self) -> None:
        """
        This method cleans the headers of the input data and adds additional columns
        """
        
        # cleans the three rows of headers to a single row
        self.data.columns = [
            ".".join([str(col).strip() for col in cols if "Unnamed" not in str(col)]) for cols in self.data.columns
        ]

        # fixes column titles
        self.data = self.data.rename(columns={"launch price": "carrier price"})
        self.data["unlocked price"] = ""
        self.data["late support ended"] = ""
        self.data["late final OS"] = ""

    def print_data(self) -> None:
        """
        This method prints the data in the dataframe
        """

        print(self.data.columns.tolist())
        for index in self.data.index:
            print(index, self.data.iloc[index].tolist())

    def clean_prices(self) -> None:
        """
        This method cleans the prices in the dataframe
        if the price is unlocked (no *), it is copied to the unlocked price column
        """
        
        for index in self.data.index:
            if index + 1 >= len(self.data):
                if not str(self.data.iloc[index, 8]).endswith("*"):
                    self.data.iat[index, 9] = self.data.iat[index, 8]
                    self.data.iat[index, 8] = ""
                break
            
            # checks if the next model is empty (if it is, there is multiple rows to handle on the current model)
            if pd.isna(self.data.iloc[index+1, 0]):
                # checks for late support ended and late final OS
                if not pd.isna(self.data.iat[index + 1, 4]):
                    self.data.iat[index, 10] = self.data.iat[index + 1, 4]
                    self.data.iat[index, 11] = self.data.iat[index + 1, 5]

                # checks for carrier price and unlocked price
                if str(self.data.iloc[index, 8]).endswith("*"):
                    unlocked_prices = self.data.iat[index+1, 8]
                    self.data.iat[index, 9] = unlocked_prices
                    self.data.drop(index + 1, inplace=True)
                else:
                    self.data.iat[index, 9] = self.data.iat[index, 8]
                    self.data.iat[index, 8] = ""
            else:
                if not str(self.data.iloc[index, 8]).endswith("*"):
                    self.data.iat[index, 9] = self.data.iat[index, 8]
                    self.data.iat[index, 8] = ""

            self.data.reset_index(drop=True, inplace=True)

    def clean_multi_values(self) -> None:
        """
        This method cleans the multi values in the dataframe
        """
        
        for index in self.data.index:
            if not pd.isna(self.data.iloc[index, 0]):
                # handles multi value models
                if " / " in str(self.data.iloc[index, 0]):
                    models = self.data.iloc[index, 0].split(" / ")
                    self.data.iat[index, 0] = models[0]
                    self.data.iat[index + 1, 0] = "iPhone " + models[1]
                    self.data.iat[index + 1, 1] = self.data.iat[index, 1]
                    self.data.iat[index + 1, 2] = self.data.iat[index, 2]
                    self.data.iat[index + 1, 3] = self.data.iat[index, 3]
                    self.data.iat[index + 1, 4] = self.data.iat[index, 4]
                    self.data.iat[index + 1, 5] = self.data.iat[index, 5]
                    self.data.iat[index + 1, 6] = self.data.iat[index, 6]
                    self.data.iat[index + 1, 7] = self.data.iat[index, 7]

                # handles multi value OS
                if " / " in str(self.data.iloc[index, 1]):
                    os_releases = self.data.iloc[index, 1].split(" / ")
                    self.data.iat[index, 1] = os_releases[0].split(" (")[0]
                    self.data.iat[index + 1, 1] = os_releases[1].split(" (")[0]

                # handles multi value release dates
                if " / " in str(self.data.iloc[index, 2]):
                    release_dates = self.data.iloc[index, 2].split(" / ")
                    self.data.iat[index, 2] = release_dates[0]
                    self.data.iat[index + 1, 2] = release_dates[1].split(" (")[0]

            if index + 1 >= len(self.data):
                break

    def format_prices(self) -> None:
        """
        This method formats the prices in the dataframe as an array of floats
        """
        
        def clean_string_number(value: str) -> float:
            """
            helper function for cleaning string numbers
            """
            
            if ":" in value:
                value = value.split(":")[1]

            return float(value)

        for index in self.data.index:
            carrier_price = self.data.iat[index, 8]
            if carrier_price:
                carrier_prices = list(map(clean_string_number, str(carrier_price).replace("$", "").replace("*", "").split("/")))
                self.data.iat[index, 8] = carrier_prices

            unlocked_price = self.data.iat[index, 9]
            if unlocked_price:
                unlocked_prices = list(map(clean_string_number, str(unlocked_price).replace("$", "").replace("*", "").split("/")))
                self.data.iat[index, 9] = unlocked_prices

    def format_dates(self) -> None:
        """
        This method formats the dates in the dataframe as pandas datetime objects
        """

        for index in self.data.index:
            release_date = self.data.iat[index, 2].split(" (")[0]
            self.data.iat[index, 2] = pd.to_datetime(release_date, format="%B %d, %Y")
            
            discontinued_date = self.data.iat[index, 3]
            if discontinued_date:
                self.data.iat[index, 3] = pd.to_datetime(discontinued_date, format="%B %d, %Y")
            
            support_ended_date = self.data.iat[index, 4]
            if support_ended_date != "current":
                self.data.iat[index, 4] = pd.to_datetime(support_ended_date, format="%B %d, %Y")

            late_support_ended_date = self.data.iat[index, 10]
            if late_support_ended_date:
                self.data.iat[index, 10] = pd.to_datetime(late_support_ended_date.split(": ")[1].split(")")[0], format="%B %d, %Y")

    def save_data(self, path: str) -> None:
        """
        This method saves the dataframe to an excel file
        """
        
        self.data.to_excel(path, index=False)


if __name__ == "__main__":
    data = pd.read_excel("SeedUnofficialAppleData.xlsx", skiprows=1, header=[0, 1, 2])
    data.replace("\xa0", " ", regex=True, inplace=True)
    pipeline = Pipeline(data)
    pipeline.set_headers()
    pipeline.clean_prices()
    pipeline.clean_multi_values()
    pipeline.format_prices()
    pipeline.format_dates()
    pipeline.save_data("CleanedData.xlsx")

    pipeline.data['carrier avg'] = pipeline.data['carrier price'].apply(lambda x: np.mean(x) if x != '' else np.nan).dropna()
    pipeline.data['unlocked avg'] = pipeline.data['unlocked price'].apply(lambda x: np.mean(x) if x != '' else np.nan).dropna()

    plt.figure(figsize=(14, 8))
    plt.plot(pipeline.data['release(d).date'], pipeline.data['carrier avg'], marker='o', label='Carrier Price Average')
    plt.plot(pipeline.data['release(d).date'], pipeline.data['unlocked avg'], marker='o', label='Unlocked Price Average')

    for i, txt in enumerate(pipeline.data['model']):
        if txt == "iPhone XS Max":
            plt.annotate(txt, (pipeline.data['release(d).date'][i], pipeline.data['carrier avg'][i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            plt.annotate(txt, (pipeline.data['release(d).date'][i], pipeline.data['unlocked avg'][i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        else:
            plt.annotate(txt, (pipeline.data['release(d).date'][i], pipeline.data['carrier avg'][i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            plt.annotate(txt, (pipeline.data['release(d).date'][i], pipeline.data['unlocked avg'][i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)

    
    def create_trendline(column: str, color: str) -> None:
        release_column_data = pipeline.data["release(d).date"]
        release_column_data = release_column_data.dropna()

        column_data = pipeline.data[column]
        column_data = column_data.dropna()

        release_column_data = release_column_data.reindex(column_data.index)
        column_data = column_data.reindex(release_column_data.index)

        coefficient = np.polyfit(range(len(release_column_data)), column_data, 1)
        polynomial = np.poly1d(coefficient)    
        best_fit_line = polynomial(range(len(release_column_data)))
        plt.plot(release_column_data, best_fit_line, label=f"{column} Trendline", linestyle='--', color=color)

    create_trendline("carrier avg", "blue")
    create_trendline("unlocked avg", "orange")

    plt.xlabel("Release Dates")
    plt.ylabel("Launch price($)")
    plt.title("iPhone Launch Carrier Prices")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
   
    # Extrapolate future prices and add to the graph
    def extrapolate_price(future_date_str):
        future_date = pd.to_datetime(future_date_str, errors='coerce')
        if pd.isna(future_date):
            print("Invalid date format. Please enter a date in the format YYYY-MM-DD.")
            return None, None

        # Calculate the number of days from the last known date
        last_known_date = release_column_data.iloc[-1]
        days_from_last_known = (future_date - last_known_date).days

        # Extrapolate the price
        future_index = len(release_column_data) + days_from_last_known
        future_price_unlocked = polynomial(future_index)
        future_price_carrier = polynomial(future_index)
        return future_date, future_price_unlocked, future_price_carrier

    # Get user input for future dates
    future_dates = ["2025-09-16", "2026-09-16", "2028-09-16"]
    future_dates_dt = []
    future_prices_unlocked = []
    future_prices_carrier = []

    for future_date in future_dates:
        future_date_dt, future_price_unlocked, future_price_carrier = extrapolate_price(future_date)
        if future_date_dt is not None:
            future_dates_dt.append(future_date_dt)
            future_prices_unlocked.append(future_price_unlocked)
            future_prices_carrier.append(future_price_carrier)

    # Combine original and future data
    all_dates = pd.concat([release_column_data, pd.Series(future_dates_dt)])
    all_prices_unlocked = pd.concat([unlocked_avg_data, pd.Series(future_prices_unlocked)])
    all_prices_carrier = pd.concat([carrier_avg_data, pd.Series(future_prices_carrier)])

    # Plot the extended data
    plt.figure(figsize=(14, 8))
    plt.plot(release_column_data, unlocked_avg_data, marker='o', linestyle='-', label='Unlocked Price Average')
    plt.plot(release_column_data, carrier_avg_data, marker='o', linestyle='-', label='Carrier Price Average')
    plt.plot(all_dates, all_prices_unlocked, marker='x', linestyle='--', color='orange', label='Extrapolated Unlocked Prices')
    plt.plot(all_dates, all_prices_carrier, marker='x', linestyle='--', color='blue', label='Extrapolated Carrier Prices')
    plt.xlabel("Release Dates")
    plt.ylabel("Launch price($)")
    plt.title("iPhone Launch Carrier Prices with Extrapolation")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()