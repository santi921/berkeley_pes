
from ase.io import read, write

def main():
    # set the chunk size
    n = 50000
    file = "../../../data/libe_chunk_0.xyz"
    read_file = read(file, index=":")
    print(len(read_file))
main()