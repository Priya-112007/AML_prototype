import os
print("Testing imports")
try:
    import pyarrow.parquet as pq
    print("pyarrow works")
except Exception as e:
    print(f"pyarrow failed: {e}")

try:
    import pandas as pd
    print("pandas works")
except Exception as e:
    print(f"pandas failed: {e}")
