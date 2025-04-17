import argparse
import sys
import inspect # To inspect function signatures
from typing import List # ADDED import for List type hint
from forge.workflows.adversarial_attack import (
    create_aa_jobs,
    run_monte_carlo_aa_jobs,
    create_aa_vasp_jobs,
    run_gradient_aa_jobs
)

# --- Helper to dynamically add subparsers ---
def add_subparser(subparsers, func, command_name):
    """Adds a subparser for a given function based on its signature."""
    # Use inspect to get function parameters and types (more robust later)
    sig = inspect.signature(func)
    
    # Create parser for the command
    parser = subparsers.add_parser(
        command_name, 
        help=func.__doc__.split('\n')[0] if func.__doc__ else f"Run {command_name}" # Simple help from docstring
    )
    
    # Add arguments based on function signature
    for name, param in sig.parameters.items():
        arg_name = f"--{name.replace('_', '-')}" # Convert snake_case to kebab-case
        kwargs = {'help': f"(Type: {param.annotation})".replace("<class '", "").replace("'>", "")}
        
        if param.default != inspect.Parameter.empty:
            kwargs['default'] = param.default
            kwargs['required'] = False
        else:
            # Assume required if no default
            kwargs['required'] = True
            
        # Handle type hints for argparse
        if param.annotation == bool:
            # Use boolean optional action for flags
            kwargs.pop('type', None) # Remove type kwarg for action
            if param.default is True:
                 # Default is True, flag should turn it False
                 parser.add_argument(f"--no-{name.replace('_', '-')}", dest=name, action='store_false', help=f"Disable {name}")
            else:
                 # Default is False, flag should turn it True
                 parser.add_argument(arg_name, dest=name, action='store_true', help=kwargs.get('help', f"Enable {name}"))
        elif param.annotation == list or param.annotation == List[str] or param.annotation == List[int]:
            kwargs['type'] = type(param.default[0]) if param.default != inspect.Parameter.empty and param.default else str # Basic type guess
            kwargs['nargs'] = '+' # One or more arguments
        elif param.annotation != inspect.Parameter.empty:
             if hasattr(param.annotation, '__origin__') and param.annotation.__origin__ == list: # Handle List[type]
                  try:
                      inner_type = param.annotation.__args__[0]
                      kwargs['type'] = inner_type
                      kwargs['nargs'] = '+'
                  except (AttributeError, IndexError):
                       kwargs['type'] = str # Fallback
                       kwargs['nargs'] = '+'
             else:
                  kwargs['type'] = param.annotation
        else:
            kwargs['type'] = str # Default to string if no annotation

        # Add argument if it wasn't handled by BooleanOptionalAction specifically
        if not (param.annotation == bool):
            parser.add_argument(arg_name, **kwargs)

    parser.set_defaults(func=func) # Associate function with parser

# --- Main CLI Function ---
def main():
    parser = argparse.ArgumentParser(description="FORGE-AA: Adversarial Attack Workflow Tools")
    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # List of workflow functions to register - ADD MORE HERE AS NEEDED
    workflow_commands = [
        create_aa_jobs,
        run_monte_carlo_aa_jobs,
        create_aa_vasp_jobs,
        run_gradient_aa_jobs
    ]

    # Dynamically create subparsers from the list
    for cmd_func in workflow_commands:
        command_name = getattr(cmd_func, 'command_name', None)
        if command_name:
             add_subparser(subparsers, cmd_func, command_name)
        else:
             print(f"Warning: Function {cmd_func.__name__} missing 'command_name' attribute.")


    # Parse arguments
    args = parser.parse_args()

    # Prepare arguments for the function call (convert back to snake_case)
    func_args = {key.replace('-', '_'): value for key, value in vars(args).items() 
                 if key not in ['command', 'func']}

    # Call the selected function
    try:
        args.func(**func_args)
    except Exception as e:
        print(f"Error executing command '{args.command}': {e}", file=sys.stderr)
        # Add more detailed error reporting or traceback if needed
        # import traceback
        # traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 