def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return "Hello, World!"

class TestClass:
    """A test class."""
    
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        """Greet with the name."""
        return f"Hello, {self.name}!"

if __name__ == "__main__":
    hello_world()
    test = TestClass("Alice")
    print(test.greet())
