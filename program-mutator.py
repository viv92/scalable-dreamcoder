import ast
import astor
import random
import string

class ProgramMutator(ast.NodeTransformer):
    def __init__(self):
        self.variable_mappings = {}
        self.transformation_probability = 0.7 # 0.3

    def visit_Name(self, node):
        # Randomly rename variables while maintaining consistency
        if isinstance(node.ctx, ast.Store):
            if random.random() < self.transformation_probability:
                new_name = ''.join(random.choices(string.ascii_lowercase, k=6))
                self.variable_mappings[node.id] = new_name
                node.id = new_name
        elif node.id in self.variable_mappings:
            node.id = self.variable_mappings[node.id]
        return node

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

# Example usage
if __name__ == "__main__":
    # Example input program
    sample_program = """
def calculate_sum(a, b):
    result = a + b
    if result > 10:
        print("Large sum!")
    return result

def main():
    x = 5
    y = 7
    print(calculate_sum(x, y))

if __name__ == "__main__":
    main()
    """
    
    # Generate mutated version
    mutated_program = mutate_program(sample_program)
    print("Original program:")
    print(sample_program)
    print("\nMutated program:")
    print(mutated_program)
