import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


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
    pipeline.print_data()
    pipeline.save_data("CleanedData.xlsx")

    # Get the first column name
    first_column_name = pipeline.data.columns[2]

    # Access the first column data
    first_column_data = pipeline.data[first_column_name]

    # Drop NaN values from the first column data
    first_column_data = first_column_data.dropna()
    
    print(first_column_data)

    # Get the launch price column name
    launch_price_column_name = pipeline.data.columns[8]

    # Access the launch price column data
    launch_price_data = pipeline.data[launch_price_column_name]

    # Drop NaN values from the launch price data
    launch_price_data = launch_price_data.dropna()

    # Extract numerical values from the launch price data
    def extract_price(price_str):
        price_str = str(price_str)
        match = re.search(r"\d+", price_str.replace(",", ""))
        return float(match.group()) if match else np.nan

    launch_price_data = launch_price_data.apply(extract_price)

    # Drop NaN values from the launch price data after extraction
    launch_price_data = launch_price_data.dropna()

    # Ensure both series have the same length by aligning indices
    first_column_data = first_column_data.reindex(launch_price_data.index)
    launch_price_data = launch_price_data.reindex(first_column_data.index)


    # Fit a linear polynomial to the data
    coefficients = np.polyfit(range(len(first_column_data)), launch_price_data, 1)
    polynomial = np.poly1d(coefficients)
    best_fit_line = polynomial(range(len(first_column_data)))

    # Plot the data
    plt.plot(first_column_data, launch_price_data, marker='o', linestyle='-', label='Launch Prices')
    plt.plot(first_column_data, best_fit_line, color="red", linestyle="--", label="Best Fit Line")
    plt.xlabel("Date")
    plt.ylabel("Launch price($)")
    plt.title("iPhone Launch Carrier Prices")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Extrapolate future prices
    # Extrapolate future prices and add to the graph
    def extrapolate_price(future_date_str):
        future_date = pd.to_datetime(future_date_str, errors='coerce')
        if pd.isna(future_date):
            print("Invalid date format. Please enter a date in the format YYYY-MM-DD.")
            return None, None

        # Calculate the number of days from the last known date
        last_known_date = first_column_data.iloc[-1]
        days_from_last_known = (future_date - last_known_date).days

        # Extrapolate the price
        future_index = len(first_column_data) + days_from_last_known
        future_price = polynomial(future_index)
        return future_date, future_price

    # Get user input for future dates
    future_dates = ["2026-09-16", "2027-09-16", "2029-09-16"]
    future_dates_dt = []
    future_prices = []

    for future_date in future_dates:
        future_date_dt, future_price = extrapolate_price(future_date)
        if future_date_dt is not None:
            future_dates_dt.append(future_date_dt)
            future_prices.append(future_price)
            

    # Combine original and future data
    all_dates = pd.concat([first_column_data, pd.Series(future_dates_dt)])
    all_prices = pd.concat([launch_price_data, pd.Series(future_prices)])

    # Plot the extended data
    plt.plot(first_column_data, launch_price_data, marker='o', linestyle='-', label='Launch Prices')
    plt.plot(all_dates, all_prices, marker='x', linestyle='--', color='green', label='Extrapolated Prices')
    plt.xlabel("Date")
    plt.ylabel("Launch price($)")
    plt.title("iPhone Launch Carrier Prices with Extrapolation")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(future_prices)