import ast
import sys
import signal
import importlib
from io import StringIO
from contextlib import contextmanager
import builtins
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd  
import os 
from tqdm import tqdm 
import re
import pickle

class TimeoutError(Exception):
    """Custom exception for timeout errors."""
    pass

@contextmanager
def timeout_handler(seconds):
    """Context manager for handling timeouts using signals."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds} seconds")

    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old handler and disable the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

class Sandbox:
    """
    A sandbox environment for executing Python code strings with:
    - Support for class definitions
    - Support for complex code structures (nested functions, classes)
    - Input arguments handling
    - Whitelisted imports
    - Custom stdout capture
    - Restricted built-ins
    - Simulated multiple user inputs
    - Execution time limits
    """
    
    SAFE_BUILTINS = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'chr', 'dict',
        'divmod', 'enumerate', 'filter', 'float', 'format', 'frozenset',
        'hash', 'hex', 'int', 'isinstance', 'issubclass', 'len', 'list',
        'map', 'max', 'min', 'oct', 'ord', 'pow', 'print', 'range',
        'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum',
        'tuple', 'zip', '__build_class__', '__name__', 'staticmethod', 
        'classmethod', 'property', 'EOFError', 'iter'
    }

    ALLOWED_IMPORTS = {
        'math', 'random', 'datetime', 'collections', 'itertools',
        'functools', 'statistics', 'decimal', 'fractions', 'numpy',
        'pandas', 'json', 're', 'time', 'bisect', 'string', 'heapq', 
        'typing', 'operator', 'copy'
    }

    def __init__(self, additional_imports: Optional[set] = None, default_timeout: int = 5):
        self.stdout = StringIO()
        self.input_values = []
        self.input_index = 0
        self.default_timeout = default_timeout
        self.allowed_imports = self.ALLOWED_IMPORTS.copy()
        if additional_imports:
            self.allowed_imports.update(additional_imports)
            
        # Initialize globals with safe built-ins
        restricted_builtins = {name: getattr(builtins, name) for name in self.SAFE_BUILTINS}
        
        # Add str.split to ensure proper string splitting behavior
        str_type = type("")
        restricted_builtins['str'] = str_type
        
        self.restricted_globals = {
            '__builtins__': restricted_builtins,
            'input': self.sandbox_input
        }
        
        # Pre-import all allowed modules
        self.setup_imports()

    def setup_imports(self):
        """Pre-import all allowed modules and setup special cases."""
        for module_name in self.allowed_imports:
            try:
                module = importlib.import_module(module_name)
                self.restricted_globals[module_name] = module
                
                # Special case for fractions.gcd
                if module_name == 'fractions':
                    from math import gcd
                    self.restricted_globals['gcd'] = gcd
                    module.gcd = gcd  # Also attach to the module itself
                
            except ImportError:
                continue

    def validate_ast(self, code_str: str) -> bool:
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            raise ValueError(f"Invalid syntax: {e}")

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.allowed_imports:
                            raise ValueError(f"Import of '{alias.name}' is not allowed")
                else:  # ImportFrom
                    if node.module not in self.allowed_imports:
                        raise ValueError(f"Import from '{node.module}' is not allowed")
                    
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ['open', 'eval', 'exec']:
                    raise ValueError(f"Function '{node.func.id}' is not allowed")
        return True

    @contextmanager
    def capture_stdout(self):
        old_stdout = sys.stdout
        sys.stdout = self.stdout
        try:
            yield
        finally:
            sys.stdout = old_stdout

    # def sandbox_input(self, prompt: str = "") -> str:
    #     """Simulates the input() function for the sandbox."""
    #     if self.input_index >= len(self.input_values):
    #         raise ValueError("Not enough input values provided for input() calls")
    #     value = self.input_values[self.input_index]
    #     self.input_index += 1
    #     return value

    # def sandbox_input(self, prompt: str = "") -> str:
    #     """Simulates the input() function for the sandbox."""
    #     if self.input_index >= len(self.input_values):
    #         raise EOFError("EOF when reading a line")
    #     value = self.input_values[self.input_index]
    #     self.input_index += 1
    #     return value

    def sandbox_input(self, prompt: str = "") -> str:
        """Simulates the input() function for the sandbox."""
        if self.input_index >= len(self.input_values):
            raise EOFError("EOF when reading a line")
            
        # Get the next input value
        value = self.input_values[self.input_index]
        self.input_index += 1
        
        # Ensure the input value is properly converted to string
        # and preserve exact whitespace formatting
        return str(value)

    def normalize_code(self, code_str: str) -> str:
        """Normalize code string by preserving relative indentation"""
        lines = code_str.strip().split('\n')
        
        # Remove empty lines at start and end while preserving internal empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
            
        # Preserve relative indentation
        cleaned_lines = []
        for line in lines:
            if line.strip():
                cleaned_lines.append(line.rstrip())
            else:
                cleaned_lines.append('')
                
        return '\n'.join(cleaned_lines)
    

    def execute(self, code_str: str, args: Optional[Dict[str, Any]] = None, 
                inputs: Optional[List[str]] = None, timeout: Optional[int] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Execute code with a timeout limit.
        
        Args:
            code_str: The code to execute
            args: Optional dictionary of arguments to pass to the code
            inputs: Optional list of input values to simulate user input
            timeout: Optional timeout in seconds (defaults to self.default_timeout)
            
        Returns:
            Tuple of (output, error)
        """
        self.stdout = StringIO()
        self.input_values = inputs or []
        self.input_index = 0
        
        timeout = timeout if timeout is not None else self.default_timeout
        
        try:
            # Normalize the code string
            code_str = self.normalize_code(code_str)
            
            # Validate code before execution
            self.validate_ast(code_str)
            
            # Create execution environment with arguments
            exec_globals = self.restricted_globals.copy()
            if args:
                exec_globals.update(args)
            
            # Compile and execute the code with timeout
            with self.capture_stdout():
                with timeout_handler(timeout):
                    code = compile(code_str, '<string>', 'exec')
                    exec(code, exec_globals)
            
            output = self.stdout.getvalue()
            return output, None
            
        except TimeoutError as e:
            return None, str(e)
        except Exception as e:
            return None, str(e)


def test_sandbox_timeout():
    sandbox = Sandbox(default_timeout=2)
    
    # Test 1: Code that completes within timeout
    fast_code = """
print("Starting fast computation...")
result = sum(range(1000000))
print(f"Result: {result}")
"""
    
    print("Testing code that should complete:")
    output, error = sandbox.execute(fast_code)
    if error:
        print(f"Error: {error}")
    else:
        print(output)

    # Test 2: Code that exceeds timeout
    slow_code = """
print("Starting slow computation...")
result = 0
for i in range(100000000):
    result += i
print(f"Result: {result}")  # Should never reach here
"""
    
    print("\nTesting code that should timeout:")
    output, error = sandbox.execute(slow_code, timeout=1)
    if error:
        print(f"Error: {error}")
    else:
        print(output)

    # Test 3: Custom timeout value
    medium_code = """
print("Starting medium computation...")
result = sum(range(10000000))
print(f"Result: {result}")
"""
    
    print("\nTesting code with custom timeout:")
    output, error = sandbox.execute(medium_code, timeout=3)
    if error:
        print(f"Error: {error}")
    else:
        print(output)

    # Test 1: Simple class
    simple_class = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        return self

p = Point(1, 2)
p.move(2, 3)
print(f"Point position: ({p.x}, {p.y})")
"""
    
    print("Testing simple class:")
    output, error = sandbox.execute(simple_class)
    if error:
        print(f"Error: {error}")
    else:
        print(output)

    # Test 2: Class with inheritance
    inheritance_class = """
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "Some sound"

class Dog(Animal):
    def speak(self):
        return "Woof!"

dog = Dog("Rex")
print(f"{dog.name} says: {dog.speak()}")
"""
    
    print("\nTesting inheritance:")
    output, error = sandbox.execute(inheritance_class)
    if error:
        print(f"Error: {error}")
    else:
        print(output)

    # Test 3: Class with static methods and properties
    advanced_class = """
class Calculator:
    PI = 3.14159
    
    @staticmethod
    def square(x):
        return x * x
    
    @classmethod
    def circle_area(cls, radius):
        return cls.PI * radius * radius

print(f"Square of 5: {Calculator.square(5)}")
print(f"Area of circle with radius 2: {Calculator.circle_area(2)}")
"""
    
    print("\nTesting advanced class features:")
    output, error = sandbox.execute(advanced_class)
    if error:
        print(f"Error: {error}")
    else:
        print(output)

    # Test 4: Using input arguments with nested functions
    nested_example = """
def create_processor(multiplier):
    def process(x):
        return x * multiplier
    return process

processor = create_processor(factor)
result = processor(input_value)
print(f"Result: {result}")
    """

    args = {
        'factor': 3,
        'input_value': 10
    }

    output, error = sandbox.execute(nested_example, args)
    if error:
        print(f"Error: {error}")
    else:
        print(output)

    
    # Test 5: Program with multiple input() calls
    input_program = """
name = input("Enter your name: ")
age = input("Enter your age: ")
color = input("Enter your favorite color: ")
print(f"Hello {name}, you are {age} years old and your favorite color is {color}.")
"""
    inputs = ["Alice", "30", "blue"]

    print("Testing program with multiple inputs:")
    output, error = sandbox.execute(input_program, inputs=inputs)
    if error:
        print(f"Error: {error}")
    else:
        print(output)



## main
if __name__ == "__main__":
    
    # load data
    data_folder = '/home/vivswan/experiments/external/deepmind-code-contests-instruct/'
    filename = 'train-hq-deduped-python.parquet'
    savename = 'filtered-dataset/filtered-hq-deduped-python.pkl'
    dataset = []
    
    fpath = data_folder + filename 
    df = pd.read_parquet(fpath, engine='pyarrow')
    r,c = df.shape 
    for i in range(r):
        row = df.iloc[i]
        lang, text = row['language'], row['text']
        if lang == 'PYTHON3':
            prompt, code = text.split('### Response')

            # process code
            code = code.splitlines()
            code = '\n'.join(code[3:-1])

            # process prompt 
            examples = prompt.split('Examples')
            if len(examples) < 2:
                examples = prompt.split('Example')
            if len(examples) < 2:
                continue 
            examples = examples[-1]
            input_pattern = r"Input\s*\n+((?:[^\n]+\n?)+?)\n{2}"
            output_pattern = r"Output\s*\n+((?:[^\n]+\n?)+?)\n{2}"
            input_examples = re.findall(input_pattern, examples)
            output_examples = re.findall(output_pattern, examples)

            # create the dataset item
            item = {}
            item['prompt'] = prompt 
            item['code'] = code 
            item['inputs'] = input_examples
            item['outputs'] = output_examples
            if len(input_examples) == len(output_examples):
                dataset.append(item)  


    filtered_dataset = []
    success_runs = 0
    correct_runs = 0

    # init sandbox 
    sandbox = Sandbox(default_timeout=2)

    # execute and test programs 
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        program = data['code']

        success = 1
        correct = 1
        # print('-' * 10)
        # print('program:\n', program)
        for j, x in enumerate(data['inputs']):

            # format input correctly
            x = x.strip()
            x = x.splitlines()
            # print('input:', x)

            output, error = sandbox.execute(program, inputs=x)
            if error:
                # print('error: ', error)
                success *= 0
                break 
            else: # check for correctness 
                gt_output = data['outputs'][j].strip()
                if output is not None:
                    output = output.strip()
                # print('execution_output:', output)
                # print('correct_output:', gt_output)
                eq = output == gt_output
                # print('eq: ', eq)
                correct *= eq 
        # print('success: ', success)
        # print('correct: ', correct)
        if success:
            success_runs += 1
            if correct:
                correct_runs += 1
                filtered_dataset.append(data)

    print('len(dataset): ', len(dataset))
    print('success_runs: ', success_runs)
    print('correct_runs: ', correct_runs)
    print('percent success runs: ', success_runs * 100 / len(dataset))
    print('percent correct runs: ', correct_runs * 100 / len(dataset))

    # Save filtered dataset using pickle
    with open(savename, 'wb') as file:
        pickle.dump(filtered_dataset, file)
