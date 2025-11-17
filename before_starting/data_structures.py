# Student should understand the basic Python data structures - lists, dictionaries, sets, and tuples. Lists and dictionaries are the most important
# These structures are everywhere in ML, from storing datasets, organizing predictions, and managing the model itself

# Lists - hold collections of item of various types 
temperatures = [72, 68, 75, 71, 69]  # A week of temperature readings
print(f"All temperatures: {temperatures}")
print(f"First day's temperature: {temperatures[0]}")  # Indexing starts at 0
print(f"Last day's temperature: {temperatures[-1]}")  # Negative indexing from end

# Adding new data
temperatures.append(73)
temperatures.append(70)
print(f"After adding another day: {temperatures}")

# Slicing - getting a subset of data (like getting just the weekend readings)
weekend = temperatures[-2:]  # Last 2 items
print(f"Weekend temperatures: {weekend}")

# List comprehension - transforming all items at once
# This pattern appears everywhere in data preprocessing
fahrenheit = [72, 68, 75]
celsius = [(f - 32) * 5/9 for f in fahrenheit]  # Convert each temperature
print(f"Converted to Celsius: {celsius}")

#2D Lists: a List made of Lists
temperatures_weekly = [[64, 69, 70, 68, 72, 73, 76], [75, 73, 71, 70, 72, 69, 67]]
print(f"Two lists inside of a list: {temperatures_weekly}")


# Dictionaries - Labeled data storage with a key:value pair. 
student_scores = {
    "Bob": 94,
    "Jim": 77,
    "Billy": 85,
    "Susan": 89,
    "Megan": 91
}
print(f"Bob's score: {student_scores["Bob"]}")

# Adding new data
student_scores["Katie"] = 99

# Getting all the keys (features)
names = list(student_scores.keys())
print(f"Names of the students: {names}")

# Safely checking if a key exists before using it:
if "Billy" in student_scores:
    print(f"Billy's score: {student_scores["Billy"]}")
    
    
# Sets - Automatically remove duplicates and allow fast checking
pets = ["cat", "dog", "cat", "bird", "dog", "cat", "fish"]
unique_pets = set(pets)
print(f"Unique pets (notice no duplicates): {unique_pets}")


# Nested Structures - Real world applications of these concepts often combine these structures
# A small dataset: each student is a dictionary, all students in a list
training_data = [
    {"study_hours": 2, "grade": 65},
    {"study_hours": 5, "grade": 78},
    {"study_hours": 8, "grade": 92},
    {"study_hours": 3, "grade": 70}
]

# Accessing a specific student's data
first_student = training_data[0]
print(f"First student studied {first_student['study_hours']} hours")

# Extracting all values for one feature (like preparing input for a model)
all_study_hours = [student["study_hours"] for student in training_data]
all_grades = [student["grade"] for student in training_data]
print(f"Study hours: {all_study_hours}")
print(f"Grades: {all_grades}")