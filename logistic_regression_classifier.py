# Import libraries.
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def main():
    bank_full = pd.read_csv("bank_full_w_dummy_vars.csv")

    # Train model on 19 binary variables.
    X = bank_full.iloc[:, 18:37]
    y = bank_full.iloc[:, 17]
    LogReg = LogisticRegression()
    LogReg.fit(X, y)

    # Creating profile of user "sam".
    new_user = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    new_user = new_user.reshape(1, -1)

    # Predicting value for user "sam" using logistic regression.
    y_pred = LogReg.predict(new_user)

    # Will user "sam" accept the special term deposit offer that the bank is soliciting? 0 if no, 1 if yes

    if y_pred == 1:
        print("The model predicts that Sam will accept the special term deposit offer that the bank is soliciting.")
    if y_pred == 0:
        print("The model predicts that Sam will NOT accept the special term deposit offer that the bank is soliciting")


if __name__ == "__main__":
    main()
