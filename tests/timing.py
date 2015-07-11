#!/usr/bin/python

# A timing script for FFTs  using clFFT and the wrappers

import sys, getopt
from subprocess import * # for popen, running processes
import os
import re # regexp package
from copy import deepcopy

# Return the line which appears just after the search string
def lineafter(searchstring, output):
    outlines = output.split('\n')
    dataline = ""
    itline = 0
    while itline < len(outlines):
        #print itline
        line = outlines[itline]
        #print line
        if re.search(searchstring, line) is not None:
            #print "\t"+str(outlines[itline])
            #print "\t"+str(outlines[itline + 1])
            dataline = outlines[itline + 1]
            itline = len(outlines)
        itline += 1

        if not dataline == "":
            #print dataline
            return dataline
    return ""

def main(argv):
    usage = '''Usage:
    \ntiming.py
    -p<string> program name
    -g<string> regexp match in line before timing output
    -P<int> OpenCL platform number
    -D<int> OpenCL device number
    -A<string> Extra arguments before main call
    -B<int> Extra arguments after main call
    -N<int> Number of tests
    -a<int> log2 of minimum test size
    -b<int> log2 of maximum test size
    -o<string> Output filename
    -d dry run
    -h display usage
    '''
    dryrun = False
    progname = ""
    platformnum = 0
    devicenum = 0
    a = 1
    b = 3
    outdir = "timing"
    N = 100
    gstring = "fft timing"
    usegpu = True
    
    prearg = []
    postarg = []
    
    try:
        opts, args = getopt.getopt(argv,"p:g:G:P:D:A:B:a:b:N:o:dh")
    except getopt.GetoptError:
        print "error in parsing arguments."
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-p"):
            progname = arg
        if opt in ("-g"):
            gstring = arg
        if opt in ("-P"):
            platformnum = int(arg)
        if opt in ("-G"):
            usegpu = int(arg) == 1
        if opt in ("-D"):
            devicenum = int(arg)
        if opt in ("-A"):
            postarg.append(arg)
        if opt in ("-B"):
            prearg.append(arg)
        if opt in ("-a"):
            a = int(arg)
        if opt in ("-b"):
            b = int(arg)
        if opt in ("-N"):
            N = int(arg)
        if opt in ("-o"):
            outdir = arg
        elif opt in ("-d"):
            dryrun = True
        elif opt in ("-h"):
            print usage
            sys.exit(0)
            
    if dryrun:
        print "Dry run!  No output actually created."

    if progname == "":
        print "please specify a program with -p"
        print usage
        sys.exit(2)
        
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = outdir + "/" + progname

    print "Output in", outfile
        
    f = open(outfile, 'wb') # erase file
    f.write("# command: timing.py " + " ".join(argv) +  "\n")
    f.write("#m\tmean\tsigma-\tsigma+\n")
    f.close()
        
    cmd0 = []
    i = 0
    while i < len(prearg):
        cmd0.append(prearg[i])
        i += 1
    cmd0.append("./" + progname)
    cmd0.append("-N" + str(N))
    if usegpu:
        cmd0.append("-P" + str(platformnum))
        cmd0.append("-D" + str(devicenum))
    print cmd0

    mstart = 2**a
    mstop = 2**b
    print "Min m: " + str(mstart) + ", max m:" +str(mstop) + "." 
    mi = a
    while mi <= b:
        m = 2 ** mi
        print mi, m
        cmd = deepcopy(cmd0)
        cmd.append("-m" + str(m))
        i = 0
        while i < len(postarg):
            cmd.append(postarg[i])
            i += 1

        print "\t" + " ".join(cmd)
        if not dryrun:
            p = Popen(cmd, stdout = PIPE, stderr = PIPE)
            p.wait()
            prc = p.returncode
            out, err = p.communicate()
            
            if (prc == 0):
                data = lineafter(gstring,  out)
                print "\t", data
                with open(outfile, "a") as myfile:
                    myfile.write(data + "\n")

            else:
                print "stdout:\n", out
                print "stderr:\n", err
        
        mi += 1
        
if __name__ == "__main__":
    main(sys.argv[1:])
