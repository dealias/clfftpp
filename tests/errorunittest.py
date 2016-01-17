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


#!/usr/bin/python -u

from subprocess import *
import sys
import getopt

usage = "Usage:\n"\
        "./errorunittest.py\n"\
        "\t-P <int>\tOpenCL platform index.\n"\
        "\t-D <int>\tOpenCL device index.\n"

testlist = ["fft1", "fft2", "fft3", "fft1r", "fft2r", "fft3r", "mfft1", "mfft1r"]

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

    for ptest in testlist:
        print ptest
        cmd = ["./" + ptest]
        cmd.append("-P" + str(P))
        cmd.append("-D" + str(D))
        print "\t", " ".join(cmd)
        p = Popen(cmd, stdout=PIPE, stderr=PIPE)
        p.wait() # sets the return code
        out, err = p.communicate() # capture output
        if not (p.returncode == 0):
            retval += 1
            print out
            print
            print err
            print
            print "\t" + ptest + " FAILED!"

    print

    if retval == 0:
        print "OK: all tests passed."
    else:
        print "Error unit test FAILED!"

if __name__ == "__main__":
    main(sys.argv[1:])
