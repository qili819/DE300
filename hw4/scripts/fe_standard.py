from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import pandas as pd

def main():
    # Load the data
    data = pd.read_csv('/tmp/cleaned_data_standard.csv')
    
    # Separate features and target
    X = data.drop('num', axis=1)
    y = data['num']

    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2))
    ])

    # Fit and transform the data
    X_transformed = pipeline.fit_transform(X)

    # Save the transformed data
    transformed_data = pd.DataFrame(X_transformed, columns=pipeline.named_steps['poly'].get_feature_names_out(X.columns))
    transformed_data['num'] = y.values
    transformed_data.to_csv('/tmp/fe_standard_data.csv', index=False)

if __name__ == "__main__":
    main()
