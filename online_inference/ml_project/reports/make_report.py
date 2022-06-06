import pandas as pd
from pandas_profiling import ProfileReport


def main():
    data = pd.read_csv("../data/raw/heart.csv")

    profile = ProfileReport(data)
    profile.to_file("report.html")


if __name__ == "__main__":
    main()
