# -*- coding: utf-8 -*-

# Import packages
import boto3
import pandas as pd




"""## Loading data"""

s3 = boto3.client('s3',
                  aws_access_key_id='ASIAYAAO5HRMJYNTOVZF',
                  aws_secret_access_key='r6wI5Abf77A6Xa9orHRq9t1EmKaCfCJXrVzj6lTH',
                  aws_session_token='IQoJb3JpZ2luX2VjEIT//////////wEaCXVzLWVhc3QtMiJHMEUCIQDkvdXTfvTCpV0/NXF0Df3vg08+5UBSFjhgkTMhuiZDGQIgEK9kUZkvWr8utCKdujT4nfcav1jmWyB0AdAhUsiLOLUq9AII3f//////////ARAAGgw1NDk3ODcwOTAwMDgiDAw4DoYeYYUvvzz+RSrIApJtDZj7lifxK5ZdZw532/0gJTi2mteWdo/WYptTj+jK6Bf19KTUCKsqLA1pPaM2ddwm+uwqX378Hn3Leqp6/MtZKvj8ne3jQok+xYobngoXgs0DMGN9IxVltYXERisVef5PLnmIKtn5kL48fSX+k80RPJb4OA7AmX99rQvc4Z99mcAwF5kCvSmRv5zlgql0CNiu+gnmteiR0rbjW0L9EOVDYbThXpW9NTAFsKN+0glVLalm+JZL4La7txlhrfbQugFYzY4mxiqy4CGOmm0sjp9tasNnI97keK5w4BbGwcPjSSdbDD4UbcweN9dkZzWADuxG2A+FoPlXlh111n62AOvhuqPCQcnDcT/t3TyeEiScuWvO6EQVkKnAJh+hNL6VjEmSsfbK60dmej9wDrZ1DCeV4VJtN8RRaj4LeYvAMwVeEz0KE4Ldn9Iw55DqsQY6pwF2n9J9rVOY5BYBptRPkRFiwnuqa24u9ge+mktnjDdS9z0ybPyf3b0yX9cPGsd51CWiyvluMacMXDwBAJexu1Eq3sIkOO4sYgA09ZFfKbZw+e+o+k0+z1NVXFxoTAFng7GV6FKRx+KUV772z98T2r/f7mjybTIjdhMhIxgSOYKngwy3t59LLOhcWmhmUTfU6zWy2qRwLsrnLchg8mP3xLGae+IJ7t9Oqg==')

bucket_name = 'de300spring2024'



def read_and_format_data(s3,bucket_name,object_key):
    try:
      data_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
      body = data_obj['Body'].read().decode('utf-8')
    except UnicodeDecodeError:
      data_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
      body = data_obj['Body'].read().decode('iso-8859-1')

    lines = body.splitlines()
    reformatted_rows = []
    current_row = []

    for line in lines:
        values = line.split()
        current_row.extend(values)
        if len(current_row) >= 76:
            reformatted_rows.append(current_row[:76])
            current_row = current_row[76:]

    column_names = [
        "patient_id", "ccf", "age", "sex", "painloc", "painexer", "relrest", "pncaden", "cp",
        "trestbps", "htn", "chol", "smoke", "cigs", "years", "fbs", "dm", "famhist", "restecg",
        "ekgmo", "ekgday", "ekgyr", "dig", "prop", "nitr", "pro", "diuretic", "proto",
        "thaldur", "thaltime", "met", "thalach", "thalrest", "tpeakbps", "tpeakbpd", "dummy",
        "trestbpd", "exang", "xhypo", "oldpeak", "slope", "rldv5", "rldv5e", "ca",
        "restckm", "exerckm", "restef", "restwm", "exeref", "exerwm", "thal", "thalsev",
        "thalpul", "earlobe", "cmo", "cday", "cyr", "num", "lmt", "ladprox", "laddist",
        "diag", "cxmain", "ramus", "om1", "om2", "rcaprox", "rcadist", "lvx1", "lvx2",
        "lvx3", "lvx4", "lvf", "cathef", "junk", "name"
    ]
    return pd.DataFrame(reformatted_rows, columns=column_names)

# List of file paths
object_keys = ['cheryl_chen/hw2/cleveland.data', 'cheryl_chen/hw2/hungarian.data', 'cheryl_chen/hw2/long-beach-va.data', 'cheryl_chen/hw2/switzerland.data']

# Read and store all DataFrames in a list
dataframes = [read_and_format_data(s3,bucket_name,object_key) for object_key in object_keys]

# Merge all DataFrames together
dt = pd.concat(dataframes, ignore_index=True)

# Save the merged DataFrame to a new CSV file, ensuring special characters are handled properly
all_data.to_csv('all_locs.csv', index=False, escapechar='\\', quoting=csv.QUOTE_ALL)

dt = pd.read_csv('all_locs.csv')






"""## Cleaning"""

dt = data[['age','sex','painloc','painexer','cp','trestbps','smoke','fbs','prop','nitr','pro','diuretic','thaldur','thalach','exang','oldpeak','slope','num']]

for column in dt.columns:
    dt[column] = pd.to_numeric(dt[column], errors='coerce')

#a. painloc, painexer: remove rows where painloc and painexer are not 1 or 0
dt = dt[dt['painloc'] >= 0]
dt.loc[dt['painloc'] > 1, 'painloc'] = 1
dt = dt[dt['painexer'] >= 0]
dt.loc[dt['painexer'] > 1, 'painexer'] = 1

#b. trestbps: replace values<100 with 0
dt.loc[dt['trestbps'] < 100, 'trestbps'] = 0

#c. oldpeak: replace values<0 with 0 and values>4 with 4
dt['oldpeak'] = pd.to_numeric(dt['oldpeak'], errors='coerce')
dt = dt[dt['oldpeak'] != -9]
dt.loc[dt['oldpeak'] < 0, 'oldpeak'] = 0
dt.loc[dt['oldpeak'] > 4, 'oldpeak'] = 4

#d. thaldur: remove rows with missing values
dt = dt[dt['thaldur'] != -9]
dt = dt[dt['thalach'] != -9]

#e. fbs, prop, nitr, pro, diuretic: remove rows with missing values
dt = dt[dt['fbs'] != -9]
dt = dt[dt['prop'] != -9]
dt.loc[dt['prop'] > 1, 'prop'] = 1
dt = dt[dt['pro'] != -9]
dt = dt[dt['diuretic'] != -9]
dt = dt[dt['nitr'] != -9]

#f. exang, slope
dt['exang'] = pd.to_numeric(dt['exang'], errors='coerce')
dt = dt[dt['exang'].isin([0,1])]
dt = dt[dt['slope'].isin([1, 2, 3])]





"""## Web scraping"""

from scrapy import Selector
import requests

import pandas as pd
import re
from typing import List

url1 = 'https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking-and-vaping/latest-release'
response = requests.get(url1)

# get the HTML file as a string
html_content = response.content
full_sel = Selector(text=html_content)

tables = full_sel.xpath('//table')

def parse_row(row:Selector) -> List[str]:
    '''
    Parses a html row into a list of individual elements
    '''
    cells = row.xpath('.//th | .//td')
    row_data = []

    for cell in cells:
        cell_text = cell.xpath('normalize-space(.)').get()
        cell_text = re.sub(r'<.*?>', ' ', cell_text)  # Remove remaining HTML tags
        # if there are br tags, there will be some binary characters
        cell_text = cell_text.replace('\xa0', '')  # Remove \xa0 characters
        row_data.append(cell_text)

    return row_data

# get the caption
tables[1].xpath('./caption/text()').get()
# get the rows
rows = tables[1].xpath('./tbody//tr')
# get header
header_row = tables[1].xpath('./thead/tr')
header = parse_row(header_row)

table_data = [parse_row(row) for row in rows]
# convert table into a data frame
percentage_by_age_df = pd.DataFrame(table_data,columns=header)
src1_age_df = percentage_by_age_df[['','2022 (%)']]
src1_age_df.rename(columns={'': 'Age', '2022 (%)': 'Rate'}, inplace=True)
src1_age_df['Rate']=src1_age_df['Rate'].astype(float)
src1_age_df['Rate']=src1_age_df['Rate']/100


url2 = 'https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm'
response = requests.get(url2)

html_content = response.content
full_sel = Selector(text=html_content)

tables = full_sel.xpath('//div[contains(@class,"card-body")]')

# get the rows
rows = tables[2].xpath('./ul/li/text()').extract()
men_rate = rows[0].split('(')[-1].strip('%)')
women_rate = rows[1].split('(')[-1].strip('%)')
data = {'Gender': ['Men', 'Women'],
        'Rate': [men_rate, women_rate]}
src2_gender_df = pd.DataFrame(data)
src2_gender_df['Rate']=src2_gender_df['Rate'].astype(float)
src2_gender_df['Rate']=src2_gender_df['Rate']/100
rows = tables[3].xpath('./ul/li/text()').extract()

def get_rates(rows):
  data = []
  for row in rows:
    rate = row.split('(')[-1].strip('%)')
    age = row.split('aged ')[1].split(' years')[0]
    data.append({'Age': age, 'Rate': float(rate)})
  rates_df = pd.DataFrame(data)
  return rates_df
src2_age_df = get_rates(rows)
src2_age_df['Rate']=src2_age_df['Rate'].astype(float)
src2_age_df['Rate']=src2_age_df['Rate']/100






"""## Imputing"""

# Make separate columns for the two sources
dt['smoke_src1']=dt['smoke']
dt['smoke_src2']=dt['smoke']

# src1 table parse age
def parse_age_range(age_range):
    if 'and over' in age_range:
        min_age = int(age_range.split(' ')[0])
        max_age = 200
    else:
        ages = age_range.split('–')
        min_age = int(ages[0])
        max_age = int(ages[1])
    return min_age, max_age

# Apply the function to each row and create new columns
src1_age_df['min_age'], src1_age_df['max_age'] = zip(*src1_age_df['Age'].apply(parse_age_range))

# Impute values with src1
# function to find age group/rate for each row
def replace_smoke(row):
    if row['smoke_src1'] == -9:
        # Find the appropriate rate based on the age range
        mask = (src1_age_df['min_age'] <= row['age']) & (src1_age_df['max_age'] >= row['age'])
        rate = src1_age_df.loc[mask, 'Rate'].values[0]
        return rate
    else:
        return row['smoke_src1']

# Apply the function to the dt DataFrame
dt['smoke_src1'] = dt.apply(replace_smoke, axis=1)


# src2 table parse age
def parse_age_range2(age_range):
    if '–' not in age_range:
        min_age = int(age_range.split(' ')[0])
        max_age = 200
    else:
        ages = age_range.split('–')
        min_age = int(ages[0])
        max_age = int(ages[1])
    return min_age, max_age

# Apply the function to each row and create new columns
src2_age_df['min_age'], src2_age_df['max_age'] = zip(*src2_age_df['Age'].apply(parse_age_range2))

# Impute values with src2
import numpy as np
men_rate = src2_gender_df[src2_gender_df['Gender']=='Men']['Rate'].iloc[0]
women_rate = src2_gender_df[src2_gender_df['Gender']=='Women']['Rate'].iloc[0]

# Impute values with src1
# function to find age group/rate for each row
def replace_smoke2(row):
    if row['smoke_src2'] == -9:
        # Find the appropriate rate based on the age range
        mask = (src2_age_df['min_age'] <= row['age']) & (src2_age_df['max_age'] >= row['age'])
        rate = src2_age_df.loc[mask, 'Rate'].values[0]
        if row['sex'] == 0:
          return rate
        else:
          return rate*men_rate/women_rate
    else:
        return row['smoke_src2']

# Apply the function to the dt DataFrame
dt['smoke_src2'] = dt.apply(replace_smoke2, axis=1)




"""## ML"""

dt['num']=(dt['num'] == 0).astype(int)


from sklearn.model_selection import train_test_split

# Assuming 'dt' is your DataFrame and 'num' is the target variable
X = dt.drop('num', axis=1)  # Features
y = dt['num']  # Target

# Split the data with 90-10 ratio and stratify on 'y' to maintain ratio of classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

# Set up the hyperparameter grid for logistic regression
log_reg_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']}
log_reg_grid = GridSearchCV(LogisticRegression(), log_reg_params, cv=5, scoring='accuracy')
log_reg_grid.fit(X_train, y_train)

# Set up the hyperparameter grid for random forest
rf_params = {'n_estimators': [10, 50, 100, 200], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [None, 10, 20, 30]}
rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)

# Check best parameters and scores
print("Best parameters for logistic regression:", log_reg_grid.best_params_)
print("Best cross-validation score for logistic regression:", log_reg_grid.best_score_)
print("Best parameters for random forest:", rf_grid.best_params_)
print("Best cross-validation score for random forest:", rf_grid.best_score_)

# Compare performance metrics and select the final model
final_model = log_reg_grid if log_reg_grid.best_score_ > rf_grid.best_score_ else rf_grid
print("Selected model type:", type(final_model.estimator).__name__)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predict on the test data
y_pred = final_model.predict(X_test)

# Evaluate the final model
print("Accuracy on test data:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))