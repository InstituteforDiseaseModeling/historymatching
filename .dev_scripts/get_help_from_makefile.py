#!/
import argparse
import os
from typing import NoReturn


def parse_help_from_make(filename: str) -> NoReturn:
    """
    Parse the help strings from makefile filename:

    Args:
        filename:

    Returns:

    """
    with open(filename, 'r') as make_in:
        lines = make_in.readlines()
        help = []
        for line in lines:
            if ": ##" in line:
                help.append(line.split(": ##"))

        help = sorted(help, key=lambda x: x[0])
        print("\n".join([f'{x[0].strip().ljust(20)}:{x[1].strip()}' for x in help]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    parse_help_from_make(f'{os.path.join(os.getcwd(), "Makefile")}')