#!/usr/bin/python

# This file is part of clFFT++.

# clFFT++ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# clFFT++ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with clFFT++.  If not, see <http://www.gnu.org/licenses/>.


from subprocess import *
import sys
import getopt
import numpy as np
from testutils import *

usage = "Usage:\n"\
        "./testfft1.py\n"\
        "\t-P <int>\tOpenCL platform index.\n"\
        "\t-D <int>\tOpenCL device index.\n" \
        "\t-m <int>\tMax problem size.\n" 

ptest = "fft1"

def main(argv):
    P = 0
    D = 0
    m = 1024
    
    try:
        opts, args = getopt.getopt(argv,"P:D:m:h")
    except getopt.GetoptError:
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-P"):
            P = int(arg)
        if opt in ("-D"):
            D = int(arg)
        if opt in ("-m"):
            m = int(arg)
        if opt in ("-h"):
            print usage
            sys.exit(0)

    retval = 0

    xlist = sizes(m)
    print xlist
            
    print ptest
    for x in xlist:
        for i in [0, 1]:
            cmd = ["./" + ptest]
            cmd.append("-P" + str(P))
            cmd.append("-D" + str(D))
            cmd.append("-x" + str(x))
            cmd.append("-i" + str(i))
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
