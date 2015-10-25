#!/usr/bin/python -u

from subprocess import *
import sys
import getopt
import numpy as np

usage = "Usage:\n"\
        "./errorunittest.py\n"\
        "\t-P <int>\tOpenCL platform index.\n"\
        "\t-D <int>\tOpenCL device index.\n"

ptest = "fft1"

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

    primes = [2, 3 ,5]

    # TODO: put a reasonable limit for short runs; otherwise use the
    # max possible
    xmax = 100000
    xlist = []

    # 2^h < xmax
    h = np.int(np.ceil( np.log(xmax) / np.log(min(primes))))
    p = np.zeros(len(primes) + 1) # TODO: this should be ints.
    
    #print "h", h
    
    while p[len(p) - 1] == 0:
        pos = 0

        #print "pos:", pos
        if p[pos] <= h:
            p[pos] += 1
        else:
            p[pos] = 0
            pos += 1
            while pos <= len(primes):
                p[pos] += 1
                if(p[pos] > h):
                    p[pos] = 0
                    pos += 1
                else:
                    break
                    
        #print p
        xval = 1
        j = 0
        while j < len(primes):
            xval *= np.int(primes[j]**p[j])
            #print primes[j], p[j], "\t"
            j += 1

        if(xval <= xmax):
            #print xval
            xlist.append(xval)

    # TODO: sort.
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
