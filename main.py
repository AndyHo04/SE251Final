import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
import re
import _tkinter as tk
from dataclasses import dataclass

from pandas.core.api import isna


class HeaderItem:
    def __init__(self, name: str, level: int, parent: "HeaderItem | None"):
        self.name: str = name
        self.level: int = level
        self.parent: "HeaderItem | None" = parent

    def get_lowest_value(self) -> "HeaderItem": ...

    def __str__(self):
        return f"Name: {self.name}, Level: {self.level}, Parent: {self.parent}"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class phoneModel:
    model_name: str
    release_date: date
    discontinued_date: date | None
    launch_prices_carrier: list[float] | None = None
    launch_prices_unlocked: list[float] | None = None


class Pipeline:
    def __init__(self, data: pd.DataFrame):
        self.data: pd.DataFrame = data
        self.headers = {}

    def set_headers(self) -> None:
        self.data.columns = [
            ".".join([str(col).strip() for col in cols if "Unnamed" not in str(col)]) for cols in self.data.columns
        ]
        self.data = self.data.rename(columns={"launch price": "carrier price"})
        self.data["unlocked price"] = ""
        self.data["late support ended"] = ""
        self.data["late final OS"] = ""

    def print_data(self) -> None:
        print(self.data.columns.tolist())
        for index in self.data.index:
            print(index, self.data.iloc[index].tolist())

    def clean_prices(self) -> None:
        for index in self.data.index:
            if index + 1 >= len(self.data):
                if not str(self.data.iloc[index, 8]).endswith("*"):
                    self.data.iat[index, 9] = self.data.iat[index, 8]
                    self.data.iat[index, 8] = ""
                break
            
            if pd.isna(self.data.iloc[index+1, 0]):
                if not pd.isna(self.data.iat[index + 1, 4]):
                    self.data.iat[index, 10] = self.data.iat[index + 1, 4]
                    self.data.iat[index, 11] = self.data.iat[index + 1, 5]

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
        for index in self.data.index:
            if not pd.isna(self.data.iloc[index, 0]):
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
                if " / " in str(self.data.iloc[index, 1]):
                    os_releases = self.data.iloc[index, 1].split(" / ")
                    self.data.iat[index, 1] = os_releases[0].split(" (")[0]
                    self.data.iat[index + 1, 1] = os_releases[1].split(" (")[0]
                if " / " in str(self.data.iloc[index, 2]):
                    release_dates = self.data.iloc[index, 2].split(" / ")
                    self.data.iat[index, 2] = release_dates[0]
                    self.data.iat[index + 1,
                                  2] = release_dates[1].split(" (")[0]

            if index + 1 >= len(self.data):
                break

    def format_prices(self) -> None:
        def clean_string_number(value: str) -> float:
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

    def save_data(self, path: str) -> None:
        self.data.to_excel(path, index=False)


if __name__ == "__main__":
    data = pd.read_excel("SeedUnofficialAppleData.xlsx",
                         skiprows=1, header=[0, 1, 2])
    data.replace("\xa0", " ", regex=True, inplace=True)
    pipeline = Pipeline(data)
    pipeline.set_headers()
    pipeline.clean_prices()
    pipeline.clean_multi_values()
    pipeline.format_prices()
    pipeline.print_data()
    pipeline.save_data("CleanedData.xlsx")
    # pipeline.clean_dates()
    # print(pipeline.get_phone_models())

    exit()

    # Get the first column name
    first_column_name = data.columns[0]

    # drop the first value of the first column
    data = data.drop(data.index[0])

    # Access the first column data
    first_column_data = data[first_column_name]

    # Drop NaN values from the first column data
    first_column_data = first_column_data.dropna()

    # Get the launch price column name
    launch_price_column_name = data.columns[-1]

    # Access the launch price column data
    launch_price_data = data[launch_price_column_name]

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

    # Ensure both series have the same length
    min_length = min(len(first_column_data), len(launch_price_data))
    first_column_data = first_column_data.iloc[:min_length]
    launch_price_data = launch_price_data.iloc[:min_length]

    # Fit a linear polynomial to the data
    coefficients = np.polyfit(
        range(len(first_column_data)), launch_price_data, 1)
    polynomial = np.poly1d(coefficients)
    best_fit_line = polynomial(range(len(first_column_data)))

    # Plot the data
    plt.plot(first_column_data, launch_price_data)
    plt.plot(
        first_column_data,
        best_fit_line,
        color="red",
        linestyle="--",
        label="Best Fit Line",
    )
    plt.xlabel("IPhone models")
    plt.ylabel("Launch price($)")
    plt.title("IPhone launch prices")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
