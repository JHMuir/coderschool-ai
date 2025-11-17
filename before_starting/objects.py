# Understanding classes is crucial because ML libraries like scikit-learn, TensorFlow,
# and keras are built around them. When you create a model, you're working with
# class instances that store data and provide methods to train and predict.

# A class is a blueprint for creating objects. Think of it like a template
# that defines what data an object holds and what actions it can perform.

class Student:
    """
    A simple Student class showing the core components:
    - __init__: Constructor that runs when creating a new student
    - Attributes: Data that each student object stores (name, grades)
    - Methods: Functions that operate on the student's data
    """
    
    def __init__(self, name, grade):
        """
        The constructor - this runs automatically when you create a Student.
        'self' refers to the specific student object being created.
        We also pass parameters which the class holds and uses internally
        """
        self.name = name      # Store the student's name
        self.grade = grade    # Store the student's grade
        
    def study(self, hours):
        """
        A method that modifies the student's data.
        Methods always have 'self' as first parameter to access the object's data.
        """
        # Studying improves the grade 
        print(f"{self.name} studied for {hours} hours. New grade: {self.grade}")
    
    def get_grade(self):
        """
        A method that returns information about the student.
        """
        return self.grade


# Creating instances (actual student objects from our blueprint)
richard = Student("Richard", 75)
sam = Student("Sam", 80)

# Each object has its own independent data
print(f"{richard.name}'s grade: {richard.grade}")
print(f"{sam.name}'s grade: {sam.grade}")

# Calling methods on specific objects
richard.study(3)  # Only Alex's grade changes
sam.study(5)   # Only Sam's grade changes

# When you use ML libraries, you do something very similar:
#
# model = ExampleModel()        # Create an instance of a model class
# model.fit(x_data, y_data)       # Call the fit method to train
# predictions = model.predict(X_test)  # Call predict method to use the model
#
# The model object stores its learned parameters just like
# our Student stores their grade. The methods (fit, predict) operate on
# that stored data.