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


import numpy as np

def sizes(maxsize):
    primes = [2, 3, 5, 7]

    list = []
    
    # 2^h < maxsize
    h = np.int(np.ceil( np.log(maxsize) / np.log(min(primes))))
    p = [0] * (len(primes) + 1)
    
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
        val = 1
        j = 0
        while j < len(primes):
            val *= np.int(primes[j]**p[j])
            #print primes[j], p[j], "\t"
            j += 1

        if(val <= maxsize):
            #print xval
            list.append(val)

    return sorted(list)

