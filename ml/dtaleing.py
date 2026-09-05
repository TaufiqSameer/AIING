import seaborn as sns;
import numpy as np;

df = sns.load_dataset('titanic');
print(df.head());
import dtale;
d = dtale.show(
    df,
    host="0.0.0.0",
    port=8000,
    open_browser=False
)

print(d._main_url)

input("Press Enter to stop D-Tale...")
