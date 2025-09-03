
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


df = pd.read_csv("calories_burn_prediction_dataset.csv")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing values per column:\n", df.isna().sum())


numeric_features = ['age','height_cm','weight_kg','bmi','duration_min','distance_km','steps',
                    'avg_heart_rate','max_heart_rate','speed_kmh','elevation_gain_m','cadence',
                    'temperature_c','humidity_percent','perceived_exertion']

categorical_features = ['user_id','gender','fitness_level','activity_type']

target = 'calories_burned'


df['bmi'] = df['weight_kg'] / ((df['height_cm']/100)**2)

df['pace_min_per_km'] = df['duration_min'] / df['distance_km'].replace(0, np.nan)
df['pace_min_per_km'].replace([np.inf, -np.inf], np.nan, inplace=True)
df['pace_min_per_km'] = df['pace_min_per_km'].fillna(df['pace_min_per_km'].median())

df['hr_zone'] = df['avg_heart_rate'] / (220 - df['age'])

df['effort_index'] = (
    0.6*df['hr_zone'].clip(0,1.2) +
    0.2*(df['perceived_exertion']/10) +
    0.1*((df['temperature_c']-22)/10) +
    0.1*((df['humidity_percent']-50)/20)
)

numeric_features += ['pace_min_per_km','hr_zone','effort_index']


gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['user_id']))

X_train = df.iloc[train_idx].drop(columns=[target])
X_test  = df.iloc[test_idx].drop(columns=[target])
y_train = df.iloc[train_idx][target]
y_test  = df.iloc[test_idx][target]


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocess = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train_transformed = preprocess.fit_transform(X_train)
X_test_transformed  = preprocess.transform(X_test)

print("Transformed train shape:", X_train_transformed.shape)
print("Transformed test shape:", X_test_transformed.shape)


joblib.dump(preprocess, "preprocess_pipeline.pkl")
pd.DataFrame(X_train_transformed).to_csv("X_train_transformed.csv", index=False)
pd.DataFrame(X_test_transformed).to_csv("X_test_transformed.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Preprocessing complete. Artifacts saved!")
