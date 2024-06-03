import boto3
import pandas as pd

def main():
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket='de300spring2024', Key='cheryl_chen/hw4/all_locs.csv')
    df = pd.read_csv(obj['Body'])
    df.to_csv('/tmp/data.csv', index=False)

