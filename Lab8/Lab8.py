import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt


dataset = pd.DataFrame({
    'Milk':     [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    'Bread':    [1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    'Eggs':     [0, 1, 1, 1, 1, 0, 1, 0, 0, 1],
    'Cheese':   [1, 1, 0, 0, 1, 1, 0, 1, 0, 0],
    'Yogurt':   [0, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    'Meat':     [1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    'Vegetables':[0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    'Fruits':   [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
})

# Використання алгоритму Apriori для знаходження частих наборів елементів
# Зниження min_support для врахування більшого набору даних
frequent_itemsets = apriori(dataset, min_support=0.3, use_colnames=True)
print("Часті набори елементів:")
print(frequent_itemsets)

# Генерація асоціативних правил
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print("\nАсоціативні правила:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Фільтрація правил на основі впевненості
strong_rules = rules[rules['confidence'] > 0.7]
print("\nСильні правила (впевненість > 0.7):")
print(strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Візуалізація асоціативних правил
plt.figure(figsize=(10, 6))
scatter = plt.scatter(rules['support'], rules['confidence'], 
                      c=rules['lift'], cmap='viridis', 
                      s=rules['lift']*20, alpha=0.7)
plt.xlabel('Підтримка')
plt.ylabel('Впевненість')
plt.title('Асоціативні правила в наборі даних продуктового магазину')
plt.colorbar(scatter, label='Підйом')
plt.grid(True, linestyle='--', alpha=0.6)

# Додавання анотацій для топ правил за підйомом
top_rules = rules.sort_values('lift', ascending=False).head(3)
for i, rule in top_rules.iterrows():
    antecedents = ', '.join([str(x) for x in list(rule['antecedents'])])
    consequents = ', '.join([str(x) for x in list(rule['consequents'])])
    plt.annotate(f"{antecedents} -> {consequents}",
                 xy=(rule['support'], rule['confidence']),
                 xytext=(10, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

plt.tight_layout()
plt.show()
