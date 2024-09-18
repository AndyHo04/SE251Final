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

    pipeline.data['carrier price'] = pipeline.data['carrier price'].apply(lambda x: x if x != '' else np.nan).dropna()
    pipeline.data['unlocked price'] = pipeline.data['unlocked price'].apply(lambda x: x if x != '' else np.nan).dropna()

    pipeline.data['carrier avg'] = pipeline.data['carrier price'].apply(lambda x: np.mean(x) if x != '' else np.nan).dropna()
    pipeline.data['unlocked avg'] = pipeline.data['unlocked price'].apply(lambda x: np.mean(x) if x != '' else np.nan).dropna()

    plt.figure(figsize=(14, 8))
    plt.plot(pipeline.data['release(d).date'], pipeline.data['carrier avg'], marker='o', label='Carrier Price Average')
    plt.plot(pipeline.data['release(d).date'], pipeline.data['unlocked avg'], marker='o', label='Unlocked Price Average')

    last_graphed = ""
    for i, txt in enumerate(pipeline.data['model']):
        if " " not in txt:
            last_graphed = txt
        elif txt.split(" ")[1][:1] == last_graphed:
            continue
        else:
            last_graphed = txt.split(" ")[1][:1]

        plt.annotate(
            txt,
            xy=(pipeline.data['release(d).date'][i], pipeline.data['unlocked avg'][i]),
            xytext=(-45, 45),
            textcoords="offset points",
            ha='center',
            fontsize=9,
            arrowprops=dict(arrowstyle='-', color='black')
        )
        plt.annotate(
            txt,
            xy=(pipeline.data['release(d).date'][i], pipeline.data['carrier avg'][i]),
            xytext=(-20, -30),
            textcoords="offset points",
            ha='center',
            fontsize=9,
            arrowprops=dict(arrowstyle='-', color='black')
        )
    
    def create_trendline(column: str, color: str):
        release_column_data = pipeline.data["release(d).date"].dropna()
        column_data = pipeline.data[column].dropna()
        release_column_data = release_column_data.reindex(column_data.index)
        column_data = column_data.reindex(release_column_data.index)

        coefficient = np.polyfit(range(len(release_column_data)), column_data, 1)
        polynomial = np.poly1d(coefficient)    
        best_fit_line = polynomial(range(len(release_column_data)))
        plt.plot(release_column_data, best_fit_line, label=f"{column.capitalize().replace(" avg", "")} Trendline", linestyle='--', color=color)
        
        return polynomial, len(release_column_data) - 1, release_column_data.iloc[-1]

    def predict_future_trendline(polynomial, last_index, last_date, num_future_points: int, color: str, column: str):
        future_indices = np.arange(last_index + 1, last_index + 1 + num_future_points)
        future_y = polynomial(future_indices)
        date_range = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=num_future_points, freq='YS')
        plt.plot(date_range, future_y, label=f"{column.capitalize().replace(" avg", "")} Future Trendline", linestyle=':', color=color)

    unlocked_trendline_polynomial, liu, ldu = create_trendline("unlocked avg", "orange")
    predict_future_trendline(polynomial=unlocked_trendline_polynomial, last_index=liu, last_date=ldu, num_future_points=5, color='orange', column='unlocked avg')
    carrier_trendline_polynomial, lic, ldc = create_trendline("carrier avg", "blue")
    predict_future_trendline(polynomial=carrier_trendline_polynomial, last_index=lic, last_date=ldc, num_future_points=5, color='blue', column='carrier avg')
    

    plt.xlabel("Release Dates")
    plt.ylabel("Launch price($)")
    plt.title("iPhone Prices")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()
