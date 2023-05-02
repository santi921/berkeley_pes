
import argparse
import json
import sys

from ase.build import niggli_reduce, minimize_tilt
from ase.io import read, write, str2bool
from asaplib.hypers import gen_default_soap_hyperparameters, gen_default_acsf_hyperparameters




def split_frames(fxyz, prefix, stride):
    # read frames
    if fxyz != 'none':
        frames = read(fxyz, ':')
        nframes = len(frames)
        print("read xyz file:", fxyz, ", a total of", nframes, "frames")

    for s in range(0, nframes, stride):
        frame = frames[s]
        niggli_reduce(frame)
        minimize_tilt(frame, order=range(0, 3), fold_atoms=True)
        write(prefix + '-' + str(s) + '.res', frame)




def acsf_hyper_gen(Zs, sharpness, scalerange, verbose, outfile):
    hypers = gen_default_acsf_hyperparameters(Zs, scalerange, sharpness, verbose)

    # output
    if outfile == 'none':
        json.dump(hypers, sys.stdout)
        print("")
    else:
        with open(outfile, 'w') as jd:
            json.dump(hypers, jd)

def soap_hyper_gen(Zs, soap_n, soap_l, multisoap, sharpness, scalerange, verbose, outfile):
    hypers = gen_default_soap_hyperparameters(Zs, multisoap=multisoap, scalerange=scalerange, soap_n=soap_n, soap_l=soap_l, sharpness=sharpness, verbose=verbose)

    # output
    if outfile == 'none':
        json.dump(hypers, sys.stdout)
        print("")
    else:
        with open(outfile, 'w') as jd:
            json.dump(hypers, jd)
