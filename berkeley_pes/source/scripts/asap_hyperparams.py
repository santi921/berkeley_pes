import asaplib, argparse
from asaplib.hypers import gen_default_soap_hyperparameters, gen_default_acsf_hyperparameters
from asaplib.io import str2bool

def hyper_handler(Zs, n, l, multisoap, sharpness, scalerange, verbose, outfile, soap):
    if soap:
        params = gen_default_soap_hyperparameters(Zs, n, l, multisoap, sharpness, scalerange, verbose)
    else:
        params = gen_default_acsf_hyperparameters(Zs, sharpness, scalerange, verbose)
    print(params)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Zs", nargs="+", type=int, help="atomic numbers to calculate descriptors for", required=True)
    parser.add_argument("--n", type=int, help="nmax for SOAP descriptors", default=6)
    parser.add_argument("--l", type=int, help="lmax for SOAP descriptors", default=6)
    parser.add_argument("--multisoap", type=int, help="How many set of SOAP descriptors do you want to use?", default=2)
    parser.add_argument("--sharpness", type=float,
                        help="sharpness factor for atom_gaussian_width, scaled to heuristic for GAP", default=1.0)
    parser.add_argument("--scalerange", type=float, help="the range of the SOAP cutoffs, scaled to heuristic for GAP",
                        default=1.0)
    parser.add_argument("--verbose", type=str2bool, nargs='?', const=True, default=False,
                        help="more descriptions of what has been done")
    parser.add_argument("--output", type=str, default='none', help="name of the output file")
    parser.add_argument("--soap", action='store_true', help="use SOAP descriptors instead of ACSF")

    args = parser.parse_args() 
    hyper_handler(args.Zs, args.n, args.l, args.multisoap, args.sharpness, args.scalerange, args.verbose, args.output, args.soap)
    #if args.soap:
    #    gen_default_soap_hyperparameters(args.Zs, args.n, args.l, args.multisoap, args.sharpness, args.scalerange, args.verbose, args.output)
    #else:
    #    gen_default_acsf_hyperparameters(args.Zs, args.sharpness, args.scalerange, args.verbose, args.output)