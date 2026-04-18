import pandas as pd
import yfinance as yf

TICKER = "GC=F"

df = yf.download(TICKER, period="max", interval="1d", auto_adjust=False)

print("\n=== RAW TYPE ===")
print(type(df))

print("\n=== RAW COLUMNS ===")
print(df.columns)

print("\n=== IS MULTIINDEX ===")
print(isinstance(df.columns, pd.MultiIndex))

print("\n=== HEAD ===")
print(df.head())

df_reset = df.reset_index()

print("\n=== COLUMNS AFTER RESET_INDEX ===")
print(df_reset.columns)

print("\n=== HEAD AFTER RESET_INDEX ===")
print(df_reset.head())

print("\n=== COLUMN NAMES AS LIST ===")
print(list(df_reset.columns))
