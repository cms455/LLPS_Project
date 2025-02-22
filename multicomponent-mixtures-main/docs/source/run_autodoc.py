#!/usr/bin/env python3

import glob
import logging
import os
import subprocess as sp

logging.basicConfig(level=logging.INFO)

OUTPUT_PATH = "packages"


def replace_in_file(infile, replacements, outfile=None):
    """reads in a file, replaces the given data using python formatting and
    writes back the result to a file.

    Args:
        infile (str):
            File to be read
        replacements (dict):
            The replacements old => new in a dictionary format {old: new}
        outfile (str):
            Output file to which the data is written. If it is omitted, the
            input file will be overwritten instead

    """
    if outfile is None:
        outfile = infile

    with open(infile) as fp:
        content = fp.read()

    for key, value in replacements.items():
        content = content.replace(key, value)

    with open(outfile, "w") as fp:
        fp.write(content)


def main(package="multicomp"):
    # remove old files
    for path in glob.glob(f"{OUTPUT_PATH}/*.rst"):
        logging.info("Remove file `%s`", path)
        os.remove(path)

    # run sphinx-apidoc
    sp.check_call(
        [
            "sphinx-apidoc",
            "--maxdepth",
            "4",
            "--output-dir",
            OUTPUT_PATH,
            "--module-first",
            f"../../{package}",  # path of the package
            f"../../{package}/tests",  # ignored path
            f"../../{package}/**/tests",  # ignored path
            f"../../{package}/conftest.py",  # ignore conftest
        ]
    )

    REPLACEMENTS = {
        "Submodules\n----------\n\n": "",
        "Subpackages\n-----------": "**Subpackages:**",
        "sim package\n===========": "Reference manual\n================",
    }

    # replace unwanted information
    for path in glob.glob(f"{OUTPUT_PATH}/*.rst"):
        logging.info("Patch file `%s`", path)
        replace_in_file(path, REPLACEMENTS)


if __name__ == "__main__":
    main()
