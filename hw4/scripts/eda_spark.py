from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when
from pyspark.ml import Transformer
import pyspark.sql.functions as F

###### Transformer Class ######
class DataCleaner(Transformer):
    def _transform(self, dataset: DataFrame) -> DataFrame:
        # Select the necessary columns
        dt = dataset.select('age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'fbs', 
                            'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 
                            'oldpeak', 'slope', 'num')

        # Convert all columns to numeric, handling errors by coercion
        for column in dt.columns:
            dt = dt.withColumn(column, col(column).cast('float'))

        dt = dt.withColumn('num', F.when(F.col('num') == 0, 0).otherwise(1))

        # a. Filter painloc and painexer to be either 1 or 0
        dt = dt.filter((col('painloc') >= 0) & (col('painexer') >= 0))
        dt = dt.withColumn('painloc', when(col('painloc') > 1, 1).otherwise(col('painloc')))
        dt = dt.withColumn('painexer', when(col('painexer') > 1, 1).otherwise(col('painexer')))

        # b. Replace trestbps < 100 with 0
        dt = dt.withColumn('trestbps', when(col('trestbps') < 100, 0).otherwise(col('trestbps')))

        # c. Replace oldpeak < 0 with 0 and oldpeak > 4 with 4
        dt = dt.withColumn('oldpeak', when(col('oldpeak') < 0, 0).when(col('oldpeak') > 4, 4).otherwise(col('oldpeak')))

        # d. Remove rows with missing thaldur and thalach
        dt = dt.filter((col('thaldur') != -9) & (col('thalach') != -9))

        # e. Remove rows with missing fbs, prop, nitr, pro, diuretic, adjusting prop to be 0 or 1
        dt = dt.filter((col('fbs') != -9) & (col('prop') != -9) & (col('nitr') != -9) &
                       (col('pro') != -9) & (col('diuretic') != -9))
        dt = dt.withColumn('prop', when(col('prop') > 1, 1).otherwise(col('prop')))

        # f. Filter exang and slope to be within specific sets
        dt = dt.filter(col('exang').isin([0, 1]))
        dt = dt.filter(col('slope').isin([1, 2, 3]))

        return dt
