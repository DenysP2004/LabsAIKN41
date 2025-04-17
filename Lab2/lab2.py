#Виведення першої букви, строки замінюючи всі букви на великі і навпаки на маленькі
s="hello world"
print(s[0])
print(s.upper()) 
print(s.lower())
#Пошук букви з якої починається слово great,розділення строки на окремі слова і з'єднання
s = "Python is great!"
print(s.find("great")) 
words= s.split(" ")
print(words)
new_s = " ".join(words)
print(new_s)

s = "Hello"
reversed_s = s[::-1] # Реверсування рядка
print(reversed_s)

s = "banana"
count = s.count("a") # Підрахунок входжень символу 'a'
print(f"Letter 'a' appears {count} times")
#заміна слова
s = "I love Python" 
new_s = s.replace("Python", "coding")
print(new_s)
#ввід строки, підрахування кількості голосних звуків
s=input("Enter string:")
vowels = "aeiouAEIOU"
count = sum(s.count(vowel) for vowel in vowels)
print(f"Number of vowels: {count}")
#перевірка строки на паліндром
if s == s[::-1]:
    print("The string is a palindrome")
else:
    print("The string is not a palindrome")
#заміна пробілів на на підкреслення
s_with_underscores = s.replace(" ", "_")
print("String with underscores:", s_with_underscores)
#Створення кількох строк і їх сортування
strings = ["Igor", "Andriy", "Zynoviy", "Denys"]
strings.sort()
print("Sorted strings:", strings)