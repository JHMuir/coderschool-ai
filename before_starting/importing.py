# Python code is organized into files (modules) and folders (packages).
# Instead of copying code everywhere, we use imports to access functionality
# from other files. ML libraries are just collections of Python files that
# we import to use their pre-built functions and classes.

import math     # Declaring we want to use the math package
# We can use a package's methods and data by declaring it like <package_name>.<thing_name>
# The dot (.) means "go inside and get something."
# Libraries are organized in hierarchies, and dots help us navigate them.
radius = 5
area = math.pi * math.pow(radius, 2) # using math's pi data and math's squaring function
print(f"Circle area: {area}")

# For example, if I say I want to use:
# my_package.my_module.my_function()
# The student should understand I'm navigating to my_package, then to my_module, and using my_function from my_module