import pandas as pd
import csv

def main():
	data = pd.read_csv('/tmp/data.csv')

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

	dt.to_csv('/tmp/cleaned_data_standard.csv', index=False)



if __name__ == "__main__":
    main()





