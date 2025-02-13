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
import astor
import random
import string

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


class InputCounter(ast.NodeVisitor):
    def __init__(self):
        self.count = 0
        
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'input':
            self.count += 1
        self.generic_visit(node)

def count_input_calls(code_str: str) -> int:
    tree = ast.parse(code_str)
    counter = InputCounter()
    counter.visit(tree)
    return counter.count


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
    

    # def execute(self, code_str: str, args: Optional[Dict[str, Any]] = None, 
    #             inputs: Optional[List[str]] = None, timeout: Optional[int] = None) -> Tuple[Optional[str], Optional[str]]:
    #     """
    #     Execute code with a timeout limit.
        
    #     Args:
    #         code_str: The code to execute
    #         args: Optional dictionary of arguments to pass to the code
    #         inputs: Optional list of input values to simulate user input
    #         timeout: Optional timeout in seconds (defaults to self.default_timeout)
            
    #     Returns:
    #         Tuple of (output, error)
    #     """
    #     self.stdout = StringIO()
    #     self.input_values = inputs or []
    #     self.input_index = 0
        
    #     timeout = timeout if timeout is not None else self.default_timeout
        
    #     try:
    #         # Normalize the code string
    #         code_str = self.normalize_code(code_str)
            
    #         # Validate code before execution
    #         self.validate_ast(code_str)
            
    #         # Create execution environment with arguments
    #         exec_globals = self.restricted_globals.copy()
    #         if args:
    #             exec_globals.update(args)
            
    #         # Compile and execute the code with timeout
    #         with self.capture_stdout():
    #             with timeout_handler(timeout):
    #                 code = compile(code_str, '<string>', 'exec')
    #                 exec(code, exec_globals)
            
    #         output = self.stdout.getvalue()
    #         return output, None
            
    #     except TimeoutError as e:
    #         return None, str(e)
    #     except Exception as e:
    #         return None, str(e)

    def execute(self, code_str: str, args: Optional[Dict[str, Any]] = None, 
            inputs: Optional[List[str]] = None, timeout: Optional[int] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Execute code with input validation and timeout limit.
        """
        self.stdout = StringIO()
        self.input_values = inputs or []
        self.input_index = 0
        
        timeout = timeout if timeout is not None else self.default_timeout
        
        try:
            code_str = self.normalize_code(code_str)
            self.validate_ast(code_str)
            
            # Count expected input calls
            expected_inputs = count_input_calls(code_str)
            if len(self.input_values) > expected_inputs:
                return None, f"Too many inputs provided. Expected {expected_inputs}, got {len(self.input_values)}"
            
            exec_globals = self.restricted_globals.copy()
            if args:
                exec_globals.update(args)
            
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



class ProgramMutator(ast.NodeTransformer):
    def __init__(self):
        self.variable_mappings = {}
        self.transformation_probability = 0.7 # 0.3

    # def visit_Name(self, node):
    #     # Randomly rename variables while maintaining consistency
    #     if isinstance(node.ctx, ast.Store):
    #         if random.random() < self.transformation_probability:
    #             new_name = ''.join(random.choices(string.ascii_lowercase, k=6))
    #             self.variable_mappings[node.id] = new_name
    #             node.id = new_name
    #     elif node.id in self.variable_mappings:
    #         node.id = self.variable_mappings[node.id]
    #     return node

    def visit_If(self, node):
        # Randomly add else clauses or modify conditions
        self.generic_visit(node)
        if not node.orelse and random.random() < self.transformation_probability:
            node.orelse = [ast.Pass()]
        return node

    def visit_BinOp(self, node):
        # Randomly modify binary operations while preserving type compatibility
        self.generic_visit(node)
        if random.random() < self.transformation_probability:
            # Group operators by compatibility to maintain program validity
            if isinstance(node.op, (ast.Add, ast.Sub)):
                # Basic arithmetic operations
                choices = [ast.Add(), ast.Sub()]
                node.op = random.choice(choices)
            elif isinstance(node.op, (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
                # Multiplicative operations
                choices = [ast.Mult(), ast.FloorDiv(), ast.Div(), ast.Mod()]
                node.op = random.choice(choices)
            elif isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
                # Bitwise operations
                choices = [ast.BitAnd(), ast.BitOr(), ast.BitXor()]
                node.op = random.choice(choices)
            elif isinstance(node.op, (ast.LShift, ast.RShift)):
                # Bit shift operations
                choices = [ast.LShift(), ast.RShift()]
                node.op = random.choice(choices)
            elif isinstance(node.op, ast.Pow):
                # Power operation (kept separate due to different computational complexity)
                if random.random() < 0.3:  # Lower probability for power to prevent huge numbers
                    choices = [ast.Mult(), ast.Pow()]
                    node.op = random.choice(choices)
        return node

    def visit_Compare(self, node):
        # Randomly modify comparison operators
        self.generic_visit(node)
        if random.random() < self.transformation_probability:
            choices = [ast.Eq(), ast.NotEq(), ast.Lt(), ast.LtE(), ast.Gt(), ast.GtE()]
            node.ops = [random.choice(choices)]
        return node

    def visit_Constant(self, node):
        # Randomly modify numeric constants
        if isinstance(node.value, (int, float)) and random.random() < self.transformation_probability:
            if isinstance(node.value, int):
                node.value = node.value + random.randint(-5, 5)
            else:
                node.value = node.value + random.uniform(-1, 1)
        return node

def mutate_program(source_code: str) -> str:
    """
    Takes a Python program as a string and returns a randomly modified version
    that remains syntactically valid.
    
    Args:
        source_code (str): Input Python program as string
    
    Returns:
        str: Modified Python program as string
    """
    try:
        # Parse the source code into an AST
        tree = ast.parse(source_code)
        
        # Apply mutations
        mutator = ProgramMutator()
        modified_tree = mutator.visit(tree)
        
        # Fix any missing locations and ensure the AST is valid
        ast.fix_missing_locations(modified_tree)
        
        # Convert back to source code
        return astor.to_source(modified_tree)
    except Exception as e:
        print(f"Error during mutation: {e}")
        return 0