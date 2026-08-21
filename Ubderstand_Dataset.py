import pandas as pd


# ============================================================
#                    PROGRAM SETTINGS
# ============================================================

FILE_PATH = "Data/EV_Battery_Dataset.csv"
LINE = "=" * 100


# ============================================================
#                    1. DATASET LOADING
# ============================================================

print()
print(LINE)
print("DATASET LOADING".center(100))
print(LINE)

df = pd.read_csv(FILE_PATH)

print("Dataset loaded successfully!")


# ============================================================
#                    2. FULL DATASET
# ============================================================

print()
print(LINE)
print("FULL DATASET".center(100))
print(LINE)

print(df)