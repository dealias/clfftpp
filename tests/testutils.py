import numpy as np

def sizes(maxsize):
    primes = [2, 3 ,5]

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

