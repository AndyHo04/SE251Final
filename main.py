import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
                item = HeaderItem(value, current_row, current_index_value)

                if item.parent == None and item.level != 0:
                    item.parent = self._get_next_possible_parent(index, item.level)

                self.headers[index] = item

            current_row += 1

        for index, value in self.headers.items():
            print(f"Index: {index}, Value: {value}")


if __name__ == "__main__":
    data = pd.read_excel("SeedUnofficialAppleData.xlsx")
    pipeline = Pipeline(data)
    pipeline.set_headers()
    value = pipeline.get_index_value(8)
    print(value)
