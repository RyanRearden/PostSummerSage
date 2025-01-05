import sage_data_client

df = sage_data_client.query(
    start="-2m", 
    filter={
        "plugin": ".*plugin-iio.*",
        "vsn": "W023"
    }
)
print(df)
print(df['value'].iloc[-1])