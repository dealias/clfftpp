#!/usr/bin/python -u

from subprocess import *
import sys

testlist = ["fft1", "fft2", "fft3", "fft1r", "fft2r", "fft3r", "mfft1", "mfft1r"]

retval = 0

for ptest in testlist:
    print ptest,
    p = Popen(['./' + ptest], stdout=PIPE, stderr=PIPE)
    p.wait() # sets the return code
    out, err = p.communicate() # capture output
    print "...done."
    if not (p.returncode == 0):
        retval += 1
        #print out
        print
        #print err
        print
        print "\t" + ptest + " FAILED!"
