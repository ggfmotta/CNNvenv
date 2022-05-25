# Load the Pandas libraries with alias 'pd' 
import pandas as pd

# dir = 'y'
mode = 'w'
# Read Experiment Dataframe
df = pd.read_excel(r'/home/gmotta/CNN/Data/Perms_Treated.xlsx', sheet_name='Dados',\
                    usecols="A:O",skiprows=range(0,2))
print(df)
df.dropna(subset=['Keq(m2)', 'K(m2)'])

# df['Image_file'] = df['Image_file']+'_'+dir

print(df.head(10))

with open('/home/gmotta/CNN/Data/AMs_data.csv', mode) as f:
    if mode == 'a':
        df.to_csv(f, header=False,index=False)
    else:
        df.to_csv(f, index=False)
        
# remember to manually remove empty rows in csv file