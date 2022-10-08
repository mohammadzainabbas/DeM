import sys

# ------------------------ #
# Helper logging functions
# ------------------------ #
def print_log(text: str) -> None:
    """
    Prints the log
    """
    print(f"[ log ]: {text}")

def print_error(text: str) -> None:
    """
    Prints the error
    """
    print(f"[ error ]: {text}")

if __name__ == "__main__":
    print_error("This is a module. Please import it.")
    sys.exit(1)
