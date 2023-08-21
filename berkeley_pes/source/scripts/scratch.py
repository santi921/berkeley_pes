from ase.io import read, write

def main():
    # Open the XYZ file
    atoms = read('path/to/your/file.xyz')

    # Select the first frame
    atoms = atoms[0]

    # Save the selected frame to a .traj file
    write('path/to/your/file.traj', atoms)
main()