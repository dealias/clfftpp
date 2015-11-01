#!/usr/bin/python -u

from subprocess import *
import sys
import getopt
import numpy as np
from testutils import *

usage = "Usage:\n"\
        "./errorunittest.py\n"\
        "\t-P <int>\tOpenCL platform index.\n"\
        "\t-D <int>\tOpenCL device index.\n"

ptest = "fft1r"

def main(argv):
    P = 0
    D = 0

    try:
        opts, args = getopt.getopt(argv,"P:D:h")
    except getopt.GetoptError:
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-P"):
            P = int(arg)
        if opt in ("-D"):
            D = int(arg)
        if opt in ("-h"):
            print usage
            sys.exit(0)

    retval = 0

    xmax = 10000
    xlist = sizes(xmax)
    print xlist
            
    print ptest
    for x in xlist:
        cmd = ["./" + ptest]
        cmd.append("-P" + str(P))
        cmd.append("-D" + str(D))
        cmd.append("-x" + str(x))
        print "\t", " ".join(cmd)
        p = Popen(cmd, stdout=PIPE, stderr=PIPE)
        p.wait() # sets the return code
        out, err = p.communicate() # capture output
        if not (p.returncode == 0):
            retval += 1
            #print out
            print
            #print err
            print
            print "\t" + ptest + " FAILED!"

    print

    if retval == 0:
        print "OK: all tests passed."
    else:
        print "Error unit test FAILED!"

    # TODO: return retval
        
if __name__ == "__main__":
    main(sys.argv[1:])
