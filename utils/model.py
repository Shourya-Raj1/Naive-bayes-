from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

def train_model(df, target, model_type):
    X = df.drop(columns=[target])
    y = df[target]

    feature_names = X.columns  # ✅ IMPORTANT

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    if model_type == "Gaussian":
        model = GaussianNB()
    else:
        model = MultinomialNB()

    model.fit(X_train, y_train)

    return model, X_test, y_test, feature_names


def predict_sample(model, input_data, feature_names):
    import pandas as pd

    df = pd.DataFrame([input_data])

    # ✅ FIX: correct column order
    df = df[feature_names]

    return model.predict(df)[0]
