# Import libraries.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def main():
    bank_full = pd.read_csv('bank_full_w_dummy_vars.csv')
    X = bank_full.iloc[:, 18:37]
    y = bank_full.iloc[:, 17]
    LogReg = LogisticRegression()
    LogReg.fit(X, y)

    y_pred = LogReg.predict(X)

    print(classification_report(y, y_pred))


if __name__ == "__main__":
    main()
