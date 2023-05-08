from ase.io import iread, write

def main():
    # set the chunk size
    n = 50000
    file = "../../../data/libe_chunk_0.xyz"
    with open(file, 'r') as input_file:
        output_file = None
        # read the xyz file iteratively
        for i, atoms in enumerate(iread(input_file)):
            # write a new file when the chunk size is reached
            if i % n == 0:
                print("chunk number: ", i // n)
                file_number = i // n
                filename_new = file.split("_")[0] + "_chunk_" + str(file_number) + ".xyz"
                if output_file is not None:
                    output_file.close()
                output_file = open(filename_new, 'w')
            write(output_file, atoms, format='xyz', append=True)

    if output_file is not None:
        output_file.close()

main()