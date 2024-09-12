import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import _tkinter as tk


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


class Pipeline:
    def __init__(self, data):
        self.data = data
        self.headers = {}

    def _get_next_possible_parent(self, index: int, level: int) -> HeaderItem:
        while index >= 0:
            node = self.headers.get(index - 1, None)
            print(node)
            while node.level >= 0:
                if node.level < level:
                    return node
                node = node.parent
            index -= 1

    def get_index_value(self, index: int) -> HeaderItem:
        return self.headers.get(index, None)

    def set_headers(self):
        more_to_fulfill = True
        current_row = 0

        while more_to_fulfill:
            data = self.data.iloc[current_row].tolist()

            if pd.isna(data[current_row]):
                more_to_fulfill = False

            for index, value in enumerate(data):
                if pd.isna(value):
                    continue

                current_index_value = self.headers.get(index, None)
                item = HeaderItem(value, current_row, None)

                item.parent = current_index_value
                if item.parent == None and item.level != 0:
                    item.parent = self._get_next_possible_parent(index, item.level)

                if current_index_value is not None:
                    self.headers[index] = item
                else:
                    self.headers[index] = item

            current_row += 1

        for index, value in self.headers.items():
            print(f"Index: {index}, Value: {value}")


if __name__ == "__main__":
    data = pd.read_excel("SeedUnofficialAppleData.xlsx")
    pipeline = Pipeline(data)
    pipeline.set_headers()
    value = pipeline.get_index_value(8)
    #print(value)
   
   # Get the first column name
    first_column_name = data.columns[0]
    
    #drop the first value of the first column
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
        match = re.search(r'\d+', price_str.replace(',', ''))
        return float(match.group()) if match else np.nan
    
    launch_price_data = launch_price_data.apply(extract_price)
    
    # Drop NaN values from the launch price data after extraction
    launch_price_data = launch_price_data.dropna()
    
    # Ensure both series have the same length
    min_length = min(len(first_column_data), len(launch_price_data))
    first_column_data = first_column_data.iloc[:min_length]
    launch_price_data = launch_price_data.iloc[:min_length]
    
    # Fit a linear polynomial to the data
    coefficients = np.polyfit(range(len(first_column_data)), launch_price_data, 1)
    polynomial = np.poly1d(coefficients)
    best_fit_line = polynomial(range(len(first_column_data)))
    
    # Plot the data
    plt.plot(first_column_data, launch_price_data)
    plt.plot(first_column_data, best_fit_line, color='red', linestyle='--', label='Best Fit Line')
    plt.xlabel("IPhone models")
    plt.ylabel("Launch price($)")
    plt.title('IPhone launch prices')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()