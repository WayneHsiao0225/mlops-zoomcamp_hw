import pandas as pd
import numpy as np 

# Q1
print(pd.__version__)

# Q2
url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv'
df = pd.read_csv(url)

num_rows = len(df)
print(f'Number of rows: {num_rows}')

# Q3
unique_brands = df['Brand'].unique()
num_brands = len(unique_brands)
print(f'Number of unique laptop brands: {num_brands}')

# Q4
missing_values = df.isnull().sum()
num_columns_with_missing = (missing_values > 0).sum()
print(f'Number of columns with missing values: {num_columns_with_missing}')

# Q5
max_dell_price = df[df['Brand'] == 'Dell']['Final Price'].max()
print(f'Maximum final price of Dell notebooks: {max_dell_price}')

# Q6
median_screen_initial = df['Screen'].median()
print(f'Initial median value of Screen: {median_screen_initial}')

most_frequent_screen = df['Screen'].mode()[0]

df['Screen'] = df['Screen'].fillna(most_frequent_screen)

median_screen_after_fillna = df['Screen'].median()
print(f'Median value of Screen after filling missing values: {median_screen_after_fillna}')

if median_screen_initial == median_screen_after_fillna:
    print("The median value has not changed.")
else:
    print("The median value has changed.")

# Q7
innjoo_laptops = df[df['Brand'] == 'Innjoo']
selected_columns = innjoo_laptops[['RAM', 'Storage', 'Screen']]
X = selected_columns.to_numpy()
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100])
w = XTX_inv @ X.T @ y
result_sum = np.sum(w)
print(f'Sum of all elements in the result w: {result_sum}')
