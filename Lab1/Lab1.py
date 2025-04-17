import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Product': ['A', 'B', 'C', 'D'],
    'Price': [ 200, 250, 333, 400],
    'Year': [2005, 1997, 2020, 2019]
})

average_price = data['Price'].mean()
min_price = data['Price'].min()
max_price = data['Price'].max()

print("Average Price:", average_price)
print("Minimum Price:", min_price)
print("Maximum Price:", max_price)

year_array = np.array(data['Year'])
average_year = np.mean(year_array)

print("Average Year:", average_year)

plt.scatter(data['Price'], data['Year'])
plt.title('Product Price and Year Comparison')
plt.xlabel('Price')
plt.ylabel('Year')
plt.show()