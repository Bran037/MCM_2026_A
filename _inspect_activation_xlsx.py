import pandas as pd
from pathlib import Path

xlsx = Path('Database') / 'Activation_Date_Phone.xlsx'
print('xlsx exists:', xlsx.exists())

xl = pd.ExcelFile(xlsx)
print('sheets:', xl.sheet_names)

for sheet in xl.sheet_names:
    df = pd.read_excel(xlsx, sheet_name=sheet)
    print('\n=== sheet:', sheet, '===')
    print('shape:', df.shape)
    print('columns:', list(df.columns))
    print(df.head(10).to_string(index=False))

# Export first sheet to CSV
first = xl.sheet_names[0]
df0 = pd.read_excel(xlsx, sheet_name=first)
out = Path('processed') / 'Activation_Date_Phone.csv'
out.parent.mkdir(exist_ok=True)
df0.to_csv(out, index=False, encoding='utf-8')
print('\nwrote', out)
