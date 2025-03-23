#!/usr/bin/env python3.12
#
# Python implementation of Hsiao sec-ded approach taken from
# his 1970 papers
# "A Class of Optimal Minimum Odd-weight-column SEC-DED Codes"
# IBM Journal of Research and Development ( Volume: 14, Issue: 4, July 1970)
#
# (c) 2025 Frédéric Pétrot <frederic.petrot@univ-grenoble-alpes.fr>
#

import sys
import numpy as np
import random

np.set_printoptions(threshold=sys.maxsize)

k = 64   # Number of bits to protect
r = 8    # Number of check bits according to theory

# Dump something looking like Verilog for the or reduction part
def dump_verilog(pcm):
    for c in range(0, r):
        print(f"c{c} = ", end='')
        start = 0;
        for d in range(0, k):
            if pcm[c,d] == 1:
                if start != 0:
                    print(" xor ", end='')
                start = 1;
                print(f"d{k - 1 - d}", end='')
        print(";")

# For the check and syndrome functions, order matters
# because the check bits are appended somewhere,
# after the lsb in our case

# Compute check bits
def compute_checkbits(pcm, v):
    check = np.zeros(shape=r, dtype=np.uint32)
    for i in range(0, r):
        for j in range(0, k):
            if pcm[i,j] == 1:
                check[i] = check[i] ^ ((v >> (k - 1 - j)) & 1)
    return check

# Compute syndrome
def compute_syndrome(pcm, v):
    check = np.zeros(shape=r, dtype=np.uint32)
    for i in range(0, r):
        for j in range(0, k + r):
            if pcm[i,j] == 1:
                check[i] = check[i] ^ ((v >> (k + r - 1 - j)) & 1)
    return check

# Make a list of 0s and 1s an integer
def l2i(l):
    v = 0
    t = len(l)
    if t != r:
        print("Argh !")
        sys.exit(1)
    for s, b in enumerate(l):
        v = v | (b << (r - 1 - s))
    # Without the int cast, strange bitwise_or error occurs!
    return int(v)

def parity(v):
    n = v & 1
    for i in range(1, k):
        n = n ^ (( v >> i) & 1)
    return int(n)

om = np.zeros(shape=(r, k + r), dtype=np.uint32)

# Nice mixture of list and numpy arrays, like always, ...
# Juste add more [] when raising a syntax error :)

# Parity check matrix (72, 64), Fig. 6 of Hsiao original '70 paper
# Obtained by hand by Zook and Dobrzynski
om[0] = [1,1,1,1,1,1,1,1, 0,0,0,0,1,1,1,1, 0,0,0,0,1,1,1,1, 0,0,0,0,1,1,0,0, 0,1,1,0,1,0,0,0, 1,0,0,0,1,0,0,0, 1,0,0,0,1,0,0,0, 1,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0]
om[1] = [1,1,1,1,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0, 1,1,1,1,0,0,1,1, 0,1,1,0,0,1,0,0, 0,1,0,0,0,1,0,0, 0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,0, 0,1,0,0,0,0,0,0]
om[2] = [0,0,1,1,0,0,0,0, 1,1,1,1,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,1,1,1,1, 0,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0, 0,0,1,0,0,0,1,0, 0,0,1,0,0,1,1,0, 0,0,1,0,0,0,0,0]
om[3] = [1,1,0,0,1,1,1,1, 0,0,0,0,0,0,0,0, 1,1,1,1,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,1, 0,0,0,1,0,0,0,1, 0,0,0,1,0,0,0,1, 0,0,0,1,0,1,1,0, 0,0,0,1,0,0,0,0]
om[4] = [0,1,1,0,1,0,0,0, 1,0,0,0,1,0,0,0, 1,0,0,0,1,0,0,0, 1,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,1,1,1,1, 0,0,0,0,0,0,0,0, 1,1,1,1,0,0,1,1, 0,0,0,0,1,0,0,0]
om[5] = [0,1,1,0,0,1,0,0, 0,1,0,0,0,1,0,0, 0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,0, 1,1,1,1,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,1,1,1,1, 0,0,0,0,1,1,0,0, 0,0,0,0,0,1,0,0]
om[6] = [0,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0, 0,0,1,0,0,0,1,0, 0,0,1,0,0,1,1,0, 1,1,0,0,1,1,1,1, 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,1,1,1,1, 0,0,0,0,0,0,1,0]
om[7] = [0,0,0,0,0,0,0,1, 0,0,0,1,0,0,0,1, 0,0,0,1,0,0,0,1, 0,0,0,1,0,1,1,0, 0,0,1,1,0,0,0,0, 1,1,1,1,0,0,0,0, 1,1,1,1,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,1]

# Easier to find which column matches in case of single error
mo = np.transpose(om)

# The nice thing is that the results do not depend on the value
# being worked on, so a rough check will do for now.
x = 0xdeadbeefcafebabe
cb = compute_checkbits(om, x)
checkbits = l2i(cb)
bv = (x << r) | checkbits
sy = compute_syndrome(om, bv)
print("Checkbits ⇒ ", cb)
print("Syndrome  ⇒ ", sy)
print("Single bit error: syndrome contains erroneous bit column pattern")
print("Double bit error: even parity indicates double error")

for i in range(0, k):
    y = x ^ (1 << i)
    sdr = compute_syndrome(om, (y << r) | checkbits)
    print(sdr, "⇒ ", end='')
    # Pattern matches erroneous bit
    w = np.where(np.all(mo == sdr, axis = 1))
    print(w[0][0])

for i in range(0, k):
    y = x ^ (3 << i)
    sdr = compute_syndrome(om, (y << r) | checkbits)
    print(sdr, parity(l2i(sdr)))

print("Testing 10000 random numbers without errors")
for _ in range(0, 1):
    n = random.getrandbits(64)
    cb = compute_checkbits(om, n)
    c = l2i(cb)
    nc = (n << r) | c
    sy = compute_syndrome(om, nc)
    s = l2i(sy)
    assert s == 0, f"Error! Syndrome is not equal to 0"

print("Testing 10000 random numbers with either simple or double errors")
for _ in range(0, 10000):
    n = random.getrandbits(64)
    cb = compute_checkbits(om, n)
    c = l2i(cb)
    nc = (n << r) | c
    bf = random.randint(0, 71)
    # Bit 64 is in position 0 in the om array
    nc = nc ^ (1 << (71 - bf))
    fb = -1
    if _ % 10 == 9:
        fb = random.randint(0, 71)
        nc = nc ^ (1 << (71 - fb))
    sy = compute_syndrome(om, nc)
    s = l2i(sy)

    if fb == bf:
        assert s == 0, f"Error! Syndrome is not equal to 0"
    else:
        assert s != 0, f"Error! Syndrome is equal to 0"
    if parity(s) == 1:
        """Single error"""
        w = np.where(np.all(mo == sy, axis = 1))
        assert fb == -1 and bf == w[0][0], "Outch! Corrected the wrong bit"
    else:
        """Double error"""
        assert fb != -1, "Arghl! WTF, Double error detected but not injected"

dump_verilog(om)
sys.exit(0)
