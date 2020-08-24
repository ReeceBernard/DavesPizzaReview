import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

df = pd.read_csv('eda_data_dropped.csv')
# Choosing the relavent columns
df_model = df[['Rating', 'yelp_rating', 'yelp_est_price','num_yelp_reviews',
		'State', 'pizza_name_yn', 'name_length', 'basketball_season_yn',
       	'football_season_yn', 'baseball_season_yn', 'season', 'Days_Since',
       	'Ten_Day_Avg', 'Zip']]

df_dum = pd.get_dummies(df_model)
X = df_dum.drop('Rating', axis=1)
y = df_dum.Rating.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multiple Linear Regression
# statsmodel regression
X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
results = model.fit().summary()
print(results)
# As you can see we are getting .48 which is very weak
#when it comes to predicting the score so it is back to the drawing board for now
