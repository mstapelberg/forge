#!/usr/bin/env python

import glob
from ase.io import read, write

def clean_xyz_files():
    # Find all xyz files in the current directory
    xyz_files = glob.glob("*.xyz")
    
    for f in xyz_files:
        try:
            # Read all frames in the file
            atoms_list = read(f, index=":")
            modified = False

            for atoms in atoms_list:
                if "virials" in atoms.info:
                    del atoms.info["virials"]
                    modified = True
                if "virial" in atoms.info:
                    del atoms.info["virial"]
                    modified = True

            if modified:
                # Overwrite the file with cleaned data
                write(f, atoms_list)
                print("Cleaned virial data from {}".format(f))
            else:
                print("No virial tags found in {}".format(f))

        except Exception as e:
            print("Error processing {}: {}".format(f, e))

if __name__ == "__main__":
    clean_xyz_files()
