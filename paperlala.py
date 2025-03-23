#!/usr/bin/env python3.12
#
# Python implementation of Lala sec-ded approach taken from
# his 2003 and 2005 papers
# (c) 2025 Frédéric Pétrot <frederic.petrot@univ-grenoble-alpes.fr>
#

import sys
import numpy as np
import random

np.set_printoptions(threshold=sys.maxsize)

k = 64   # Number of bits to protect
r = 8    # Number of check bits according to theory

# Dump something looking like Verilog
def dump_verilog(pcm):
    for c in range(0, r + 2):
        if c < r:
            print(f"c{c} = ", end='')
        else:
            print(f"m{c-r} = ", end='')
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
    check = np.zeros(shape=r + 2, dtype=np.uint32)
    for i in range(0, r + 2):
        for j in range(0, k):
            if pcm[i,j] == 1:
                check[i] = check[i] ^ ((v >> (k - 1 - j)) & 1)
    return check

# Compute syndrome
def compute_syndrome(pcm, v):
    check = np.zeros(shape=r + 2, dtype=np.uint32)
    for i in range(0, r + 2):
        for j in range(0, k + r + 2):
            if pcm[i,j] == 1:
                check[i] = check[i] ^ ((v >> (k + r + 2 - 1 - j)) & 1)
    return check

# Make a list of 0s and 1s an integer
def l2i(l):
    v = 0
    # t = len(l)
    # assert t = r + 2, "Argh !"
    for s, b in enumerate(l):
        v = v | (b << (r + 2 - 1 - s))
    return int(v)

def parity(v):
    n = v & 1
    for i in range(1, k):
        n = n ^ (( v >> i) & 1)
    return int(n)

# This error checking function seems to cost quite a lot, ...
# We need mod-3 on 8 bits, popcnt on 8 bits, and lots of
# small comparators
# This is reversed engineered from the syndrome value for 64-bit only.
# Lala's papers are either erroneous, or I didn't understand the checking procedure ...
def check_error(syndrome):
    # Remember, the 2 lsb are m1 and m0
    s = syndrome >> 2
    m10 = syndrome & 3
    m10p = s%3

    sb = np.array(list(np.binary_repr(s).zfill(8))).astype(np.int8)
    m10b = np.array(list(np.binary_repr(m10).zfill(2))).astype(np.int8)
    m10pb = np.array(list(np.binary_repr(m10p).zfill(2))).astype(np.int8)

    # print(f"Syndrome: ", sb, m10b, m10pb, f"({s.bit_count()})", f"{np.where(np.all(mo[:,0:8] == sb, axis = 1))[0][0]:2d} ⇒ ", end='')
    print(f"Syndrome: ", sb, m10b, m10pb, f"({s.bit_count()})", end='')

    if syndrome == 0:
        print("""No error""")
        return 0
    if (m10 == 1 or m10 == 2) and s == 0: # m1 xor m0
        print("""Residue bit error""")
        return 0
    if m10 == 0 and s.bit_count() == 1:
        print("""Check bit error""")
        return 2
    #if (m10 == 0 and m10p == 0 and s.bit_count() == 3) or (m10 == 1 and m10p == 1 and s.bit_count() == 4) or (m10 == 2 and m10p == 2 and s.bit_count() == 2):
    if (m10 == 0 and s.bit_count() == 3) or (m10 == 1 and m10p != 3 and s.bit_count() == 2):
        print("""Single bit error""")
        return 3
    print("""Double bit error""")
    return 4

# First Lala paper, as many ones (216) as in Hsiao codes, so not really interesting for 64-bit values
# "A Single Error Correcting and Double Error Detecting Coding Scheme for Computer Memory Systems"
# Proceedings of the 18th IEEE International Symposium on Defect and Fault Tolerance in VLSI Systems (DFT’03)
pm = np.zeros(shape=(r + 2, k + r + 2), dtype=np.uint32)
pm[0] = [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0,0,]
pm[1] = [1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0,1,0,0,0,0,0,0,0,0,]
pm[2] = [0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,]
pm[3] = [0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0, 0,0,0,1,0,0,0,0,0,0,]
pm[4] = [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,1,1,0,0,0,0,1,0,0,0,1,0,0,0,1,1,1,0,0,0, 0,0,0,0,1,0,0,0,0,0,]
pm[5] = [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0, 0,0,0,0,0,1,0,0,0,0,]
pm[6] = [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,1,0,1,0,1, 0,0,0,0,0,0,1,0,0,0,]
pm[7] = [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1, 0,0,0,0,0,0,0,1,0,0,]
# Mod-3 of the 8 above bits considered as an int, top bit msb 
pm[8] = [0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0, 0,0,0,0,0,0,0,0,1,0,]
pm[9] = [0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,1,0,1, 0,0,0,0,0,0,0,0,0,1,]
mp = np.transpose(pm)

# "Single error correcting and double error detecting coding scheme",
# P.K. Lala, P. Thenappan and M.T. Anwar,
# IEE ELECTRONICS LETTERS 23rd June 2005 Vol. 41 No. 13
# Now only 202 ones, which is 14 less than Hsiao codes
om = np.zeros(shape=(r + 2, k + r + 2), dtype=np.uint32)
om[0] = [1,0,0,0,1,1,0,1, 0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0, 0,0,1,0,0,0,0,1, 0,0,1,0,1,0,1,0, 0,1,1,1,1,0,0,1, 0,1,1,1,0,0,0,0, 1,0,1,0,1,0,0,0, 1,0,0,0,0,0,0,0,0,0]
om[1] = [0,0,0,0,0,0,0,1, 1,1,0,0,0,0,1,0, 1,0,1,0,1,0,0,0, 0,0,0,0,0,0,0,1, 1,0,1,0,0,0,1,1, 1,0,1,0,0,0,0,0, 1,1,0,0,1,0,1,0, 0,0,0,0,1,1,1,0, 0,1,0,0,0,0,0,0,0,0]
om[2] = [0,0,0,1,0,1,0,0, 0,0,0,0,0,1,1,0, 0,0,0,0,0,0,1,1, 0,0,0,1,0,1,0,0, 0,0,0,1,0,0,0,0, 0,0,0,0,0,0,0,1, 0,0,0,0,1,1,1,1, 0,1,1,1,1,0,1,0, 0,0,1,0,0,0,0,0,0,0]
om[3] = [1,0,0,0,0,0,0,0, 0,1,1,1,0,0,0,0, 0,0,0,0,0,1,0,1, 1,0,0,0,0,1,0,0, 0,1,0,1,0,1,0,1, 1,1,0,1,1,1,0,1, 1,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,1, 0,0,0,1,0,0,0,0,0,0]
om[4] = [0,0,1,1,0,0,1,0, 0,0,0,0,0,0,0,1, 0,0,1,1,0,0,0,0, 1,0,0,0,1,1,1,1, 1,0,0,0,1,1,0,0, 0,0,0,0,1,1,1,0, 1,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,0, 0,0,0,0,1,0,0,0,0,0]
om[5] = [0,1,0,0,0,0,1,0, 0,0,0,0,0,0,0,0, 1,1,0,0,0,1,0,0, 0,0,1,1,0,0,1,0, 0,1,1,0,1,0,0,0, 0,1,0,0,0,1,1,0, 0,0,1,0,0,1,0,0, 0,0,0,0,0,0,1,1, 0,0,0,0,0,1,0,0,0,0]
om[6] = [0,0,1,0,0,0,0,0, 1,0,0,1,1,1,0,0, 0,1,0,0,0,0,0,0, 0,1,0,0,1,0,0,0, 0,0,0,1,0,1,1,0, 1,0,0,1,0,0,1,0, 0,0,0,1,1,0,0,0, 1,1,0,1,0,1,0,1, 0,0,0,0,0,0,1,0,0,0]
om[7] = [0,1,0,0,1,0,0,0, 0,0,1,0,0,0,0,1, 0,0,0,0,1,0,1,0, 0,1,0,0,1,0,1,0, 1,1,0,0,0,0,0,1, 0,0,1,0,0,0,0,0, 0,0,1,1,0,1,1,1, 0,0,1,1,0,1,0,0, 0,0,0,0,0,0,0,1,0,0]
                                                                                                                                
om[8] = [0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,1,0]
om[9] = [1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,1]

# Easier to find which column matches in case of single error
mo = np.transpose(om)

x = 0x123456789abcdef0
cb = compute_checkbits(om, x)
checkbits = l2i(cb)
bv = (x << r + 2) | checkbits
sy = compute_syndrome(om, bv)
# print("Checkbits ⇒ ", cb)
# print("Syndrome  ⇒ ", sy)
check_error(l2i(sy))

"""
print("Testing 10000 random numbers without errors")
for _ in range(0, 1):
    n = random.getrandbits(64)
    cb = compute_checkbits(om, n)
    c = l2i(cb)
    nc = (n << r + 2) | c
    sy = compute_syndrome(om, nc)
    s = l2i(sy)
    assert s == 0, f"Error! Syndrome is not equal to 0"

# Introduce single bit error
# First in data
print("Single bit error: syndrome contains erroneous bit column pattern")
for x in [0x0, 0x34, 0x136, 0x5378, 0x9ef3a, 0xffffff, 0x123456789abcdef0, 0x0123456789abcdef, 0xabcdef0123456789, 0xdeadbeefcafebabe]:
    cb = compute_checkbits(om, x)
    checkbits = l2i(cb)
    for i in range(0, k):
        y  = x ^ (1 << i)
        bv = (y << r + 2) | checkbits
        sy = compute_syndrome(om, bv)
        s = l2i(sy[0:8])
        # print(f"Syndrome  ⇒ ", sy[0:8], sy[8:10], "     ", f"({s.bit_count()})", np.where(np.all(mo[:,0:8] == sy[0:8], axis = 1))[0])
        # print(f"{l2i(sy):010b}")
        check_error(l2i(sy))

# Then in syndrome
print("Single bit error: syndrome contains erroneous bit column pattern")
for x in [0x0, 0x34, 0x136, 0x5378, 0x9ef3a, 0xffffff, 0x123456789abcdef0, 0x0123456789abcdef, 0xabcdef0123456789, 0xdeadbeefcafebabe]:
    cb = compute_checkbits(om, x)
    checkbits = l2i(cb)
    for i in range(0, r + 2):
        cx  = checkbits ^ (1 << i)
        bv = (x << r + 2) | cx
        sy = compute_syndrome(om, bv)
        s = l2i(sy[0:8])
        # print(f"Syndrome  ⇒ ", sy[0:8], sy[8:10], "     ", f"({s.bit_count()})", np.where(np.all(mo[:,0:8] == sy[0:8], axis = 1))[0])
        # print(f"{l2i(sy):010b}")
        check_error(l2i(sy))


print("Testing 10000 random numbers with double errors")
for _ in range(0, 10000):
    n = random.getrandbits(64)
    cb = compute_checkbits(om, n)
    c = l2i(cb)
    nc = (n << r + 2) | c
    bf = random.randint(0, 73)
    nc = nc ^ (1 << (73 - bf))
    fb = random.randint(0, 73)
    nc = nc ^ (1 << (73 - fb))
    sy = compute_syndrome(om, nc)
    s = l2i(sy)
    if bf != fb:
        check_error(s)
"""

x = 0x123456789abcdef0
for i in range(0, k + r + 1):
    for j in range(i + 1, k + r + 2):
        bv = (x << r + 2) | checkbits
        bv  = bv ^ (1 << i)
        bv  = bv ^ (1 << j)
        sy = compute_syndrome(om, bv)
        s = l2i(sy[0:8])
        # print(f"Syndrome  ⇒ ", sy[0:8], sy[8:10], "     ", f"({s.bit_count()})", np.where(np.all(mo[:,0:8] == sy[0:8], axis = 1))[0])
        # print(f"{l2i(sy):010b}")
        check_error(l2i(sy))


"""
# Same as all sec-ded codes, the number of the column that matches
# is the bit position to flip
print("Single bit error: syndrome contains erroneous bit column pattern")
for i in range(0, k):
    y = x ^ (0x1 << i)
    sdr = compute_syndrome(om, (y << r + 2) | checkbits)
    print(sdr, "⇒ ", end='')
    # Pattern matches erroneous bit
    w = np.where(np.all(mo == sdr, axis = 1))
    print(w[0])

print("Double bit error: checking procedure isn't that easy, I am afraid!")
for i in range(0, k):
    y = x ^ (0x3 << i)
    sdr = compute_syndrome(om, (y << r + 2) | checkbits)
    print(sdr, parity(l2i(sdr)))
"""

sys.exit(0)
