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
    print("module compute_checkbits (\n",
          "    input  wire [63:0] d,\n",
          "    output wire  [8:0] c\n);\n")

    for c in range(0, r + 1):
        print(f"assign c[{c}] = ", end='')
        start = 0;
        for d in range(0, k):
            if pcm[c,d] == 1:
                if start != 0:
                    print(" ^ ", end='')
                start = 1;
                print(f"d[{k - 1 - d}]", end='')
        print(";")
    print("endmodule")

    print("module compute_syndrome (\n",
          "    input  wire [63:0] d,\n",
          "    input  wire  [8:0] c,\n",
          "    output wire  [8:0] s\n);\n")

    for s in range(0, r + 1):
        print(f"assign s[{s}] = ", end='')
        start = 0;
        for d in range(0, k + r + 1):
            if pcm[s,d] == 1:
                if start != 0:
                    print(" ^ ", end='')
                start = 1;
                if d < k:
                    print(f"d[{k - 1 - d}]", end='')
                else:
                    print(f"c[{s}]", end='')
        print(";")
    print("endmodule")

    print("module check_syndrome (\n",
          "    input  wire  [8:0] syndrome,\n",
          "    output wire        ne, // no error\n",
          "    output wire        ce, // single checkbit error\n",
          "    output wire        se, // single bit error\n",
          "    output wire        de, // double bit error\n",
          ");\n")

    print(f"wire [7:0] s = syndrome[8:1];")
    print(f"wire [4:0] p = s[7] + s[6] + s[5] + s[4] + s[3] + s[2] + s[1] + s[0];")
    print(f"wire       m = syndrome[0];")
    print(f"assign ne = syndrome == 0;")
    print(f"assign ce = (m == 0 && p == 1) || (m == 1 && s == 0);")
    print(f"assign se = (m == 0 && p == 3) || (m == 1 && p == 2);")
    print(f"assign de =  ~(ne | ce | se);")
    print("endmodule")


# For the check and syndrome functions, order matters
# because the check bits are appended somewhere,
# after the lsb in our case

# Compute check bits
def compute_checkbits(pcm, v):
    check = np.zeros(shape=r + 1, dtype=np.uint32)
    for i in range(0, r + 1):
        for j in range(0, k):
            if pcm[i,j] == 1:
                check[i] = check[i] ^ ((v >> (k - 1 - j)) & 1)
    return check

# Compute syndrome
def compute_syndrome(pcm, v):
    check = np.zeros(shape=r + 1, dtype=np.uint32)
    for i in range(0, r + 1):
        for j in range(0, k + r + 1):
            if pcm[i,j] == 1:
                check[i] = check[i] ^ ((v >> (k + r + 1 - 1 - j)) & 1)
    return check

# Make a list of 0s and 1s an integer
def l2i(l):
    v = 0
    t = len(l)
    # assert t = r + 1, "Argh !"
    for s, b in enumerate(l):
        v = v | (b << (t - 1 - s))
    return int(v)

def parity(v):
    n = v & 1
    for i in range(1, k):
        n = n ^ (( v >> i) & 1)
    return int(n)

# Using a check without mod3 works when protecting below 84 bits,
# which is our case.
# We still need popcount on 8 bits, which seems acceptable
def check_error(syndrome):
    s = syndrome >> 1
    m = syndrome & 1

    sb = np.array(list(np.binary_repr(s).zfill(8))).astype(np.int8)
    mb = np.array(list(np.binary_repr(m).zfill(1))).astype(np.int8)

    # print(f"Syndrome: ", sb, mb, f"({s.bit_count()})", end='')

    if syndrome == 0:
        # print("""No error""")
        return 0
    if m == 1 and s == 0:
        # print("""Residue bit error""")
        return 1
    if m == 0 and s.bit_count() == 1:
        # print("""Check bit error""")
        return 2
    if (m == 0 and s.bit_count() == 3) or (m == 1 and s.bit_count() == 2):
        # print("""Single bit error""")
        return 3
    # print("""Double bit error""")
    return 4

# 

# "Single error correcting and double error detecting coding scheme",
# P.K. Lala, P. Thenappan and M.T. Anwar,
# IEE ELECTRONICS LETTERS 23rd June 2005 Vol. 41 No. 13
# Now only 202 ones, which is 14 less than Hsiao codes
om = np.zeros(shape=(r + 1, k + r + 1), dtype=np.uint32)
#        d d d d d d d d  d d d d d d d d  d d d d d d d d  d d d d  d d d d  d d d d d d d d  d d d d d d d d  d d d d d d d d  d d d d d d d d 
#        6 6 5 5 5 5 5 5  5 5 5 5 5 4 4 4  4 4 4 4 4 4 4 3  3 3 3 3  3 3 3 3  3 3 2 2 2 2 2 2  2 2 2 2 1 1 1 1  1 1 1 1 1 1 0 0  0 0 0 0 0 0 0 0 
#        3 2 1 9 8 7 6 5  4 3 2 1 0 0 9 8  7 6 5 4 3 2 1 0  9 8 7 6  5 4 3 2  1 0 9 8 7 6 5 4  3 2 1 0 9 8 7 6  5 4 3 2 1 0 9 8  7 6 5 4 3 2 1 0  8 7 6 5 4 3 2 1 0 

om[0] = [1,0,0,0,1,1,0,1, 0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0, 0,0,1,0, 0,0,0,1, 0,0,1,0,1,0,1,0, 0,1,1,1,1,0,0,1, 0,1,1,1,0,0,0,0, 1,0,1,0,1,0,0,0, 1,0,0,0,0,0,0,0,0] #23
om[1] = [0,0,0,0,0,0,0,1, 1,1,0,0,0,0,1,0, 1,0,1,0,1,0,0,0, 0,0,0,0, 0,0,0,1, 1,0,1,0,0,0,1,1, 1,0,1,0,0,0,0,0, 1,1,0,0,1,0,1,0, 0,0,0,0,1,1,1,0, 0,1,0,0,0,0,0,0,0] #23
om[2] = [0,0,0,1,0,1,0,0, 0,0,0,0,0,1,1,0, 0,0,0,0,0,0,1,1, 0,0,0,1, 0,1,0,0, 0,0,0,1,0,0,0,0, 0,0,0,0,0,0,0,1, 0,0,0,0,1,1,1,1, 0,1,1,1,1,0,1,0, 0,0,1,0,0,0,0,0,0] #20
om[3] = [1,0,0,0,0,0,0,0, 0,1,1,1,0,0,0,0, 0,0,0,0,0,1,0,1, 1,0,0,0, 0,1,0,0, 0,1,0,1,0,1,0,1, 1,1,0,1,1,1,0,1, 1,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,1, 0,0,0,1,0,0,0,0,0] #22
om[4] = [0,0,1,1,0,0,1,0, 0,0,0,0,0,0,0,1, 0,0,1,1,0,0,0,0, 1,0,0,0, 1,1,1,1, 1,0,0,0,1,1,0,0, 0,0,0,0,1,1,1,0, 1,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,0, 0,0,0,0,1,0,0,0,0] #22
om[5] = [0,1,0,0,0,0,1,0, 0,0,0,0,0,0,0,0, 1,1,0,0,0,1,0,0, 0,0,1,1, 0,0,1,0, 0,1,1,0,1,0,0,0, 0,1,0,0,0,1,1,0, 0,0,1,0,0,1,0,0, 0,0,0,0,0,0,1,1, 0,0,0,0,0,1,0,0,0] #19
om[6] = [0,0,1,0,0,0,0,0, 1,0,0,1,1,1,0,0, 0,1,0,0,0,0,0,0, 0,1,0,0, 1,0,0,0, 0,0,0,1,0,1,1,0, 1,0,0,1,0,0,1,0, 0,0,0,1,1,0,0,0, 1,1,0,1,0,1,0,1, 0,0,0,0,0,0,1,0,0] #22
om[7] = [0,1,0,0,1,0,0,0, 0,0,1,0,0,0,0,1, 0,0,0,0,1,0,1,0, 0,1,0,0, 1,0,1,0, 1,1,0,0,0,0,0,1, 0,0,1,0,0,0,0,0, 0,0,1,1,0,1,1,1, 0,0,1,1,0,1,0,0, 0,0,0,0,0,0,0,1,0] #22
om[8] = [1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1, 0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,1] #29

# Easier to find which column matches in case of single error
mo = np.transpose(om)


dump_verilog(om)
sys.exit(1)

def dump(pcm):
    for i in range(0, k + r + 1):
        row = pcm[i,:] 
        rl = row.tolist()
        x = l2i(rl)
        print(f"{x:09b}")
    sys.exit(1)

# Check that the matrix does not contain twice the same value
# Crap but since this is a very small array, we'll live with that
def check_duplicate_rows(pcm):
    for i in range(0, k + r + 1):
        row = pcm[i,0:8] 
        rl = row.tolist()
        for j in range(i + 1, k + r + 1):
            wor = pcm[j,0:8]
            wl = wor.tolist()
            if rl == wl:
                print("Duplicate !", f"row {i} {row} = row {j} {pcm[j,0:8]}")
    sys.exit(1)

# Check that the matrix does not contain twice the same value
# Crap but since this is a very small array, we'll live with that
def check_xor_rows(pcm):
    print("Checking rows for xor")
    for i in range(0, k + r + 1):
        row = pcm[i,:] 
        rl = row.tolist()
        rr = l2i(rl)
        # print(f'rr:{rr:09b}')
        for j in range(i + 1, k + r + 1):
            wor = pcm[j,:]
            wl = wor.tolist()
            ww = l2i(wl)
            # print(f'ww:{ww:09b}')
            xx = rr ^ ww
            if xx == 0:
                print("Xor error !", f"row {i} {row} xor row {j} {wor} = 0")
            for kk in range(0, k + r + 1):
                orw = pcm[kk,:] 
                ol = orw.tolist()
                oo = l2i(ol)
                # print(f'oo:{oo:09b}')
                if xx ^ oo == 0:
                    print("Xor error !", f"row {i} {row} xor row {j} {wor} xor row {kk} {orw}")
    sys.exit(1)

#check_xor_rows(mo)

x = 0x123456789abcdef0
cb = compute_checkbits(om, x)
checkbits = l2i(cb)
bv = (x << (r + 1)) | checkbits
sy = compute_syndrome(om, bv)
# print("Checkbits ⇒ ", cb)
# print("Syndrome  ⇒ ", sy)
check_error(l2i(sy))

def check_no_error():
    print("Testing 100 random numbers without errors")
    for _ in range(0, 100):
        x = random.getrandbits(64)
        cb = compute_checkbits(om, x)
        c = l2i(cb)
        nc = (x << (r + 1)) | c
        sy = compute_syndrome(om, nc)
        s = l2i(sy)
        assert check_error(s) == 0, "Syndrome not 0 while injecting no error! Dying ..."

# Introduce single bit error
# First in data
def check_single_bit_data_error():
    print("Testing 100 random numbers with Single bit data error: syndrome contains erroneous bit column pattern")
    for _ in range(0, 100):
        x = random.getrandbits(64)
        cb = compute_checkbits(om, x)
        checkbits = l2i(cb)
        for i in range(0, k):
            y  = x ^ (1 << i)
            bv = (y << (r + 1)) | checkbits
            sy = compute_syndrome(om, bv)
            s = l2i(sy[0:8])
            assert check_error(l2i(sy)) == 3, "Single bit data error uncorreclty classified as something else !"

# Then in syndrome
def check_single_bit_checkbits_error():
    print("Testing 100 random numbers with Single bit checkbits error: contains erroneous bit column pattern")
    for _ in range(0, 100):
        x = random.getrandbits(64)
        cb = compute_checkbits(om, x)
        checkbits = l2i(cb)
        for i in range(0, r + 1):
            cx  = checkbits ^ (1 << i)
            bv = (x << (r + 1)) | cx
            sy = compute_syndrome(om, bv)
            s = l2i(sy[0:8])
            check_error(l2i(sy))
            assert check_error(l2i(sy)) == 1 or check_error(l2i(sy)) == 2 , "Single bit checkbits error uncorreclty classified as something else !"

"""
print("Testing 10000 random numbers with double errors")
for _ in range(0, 10000):
    n = random.getrandbits(64)
    cb = compute_checkbits(om, n)
    c = l2i(cb)
    nc = (n << (r + 1)) | c
    bf = random.randint(0, 73)
    nc = nc ^ (1 << (73 - bf))
    fb = random.randint(0, 73)
    nc = nc ^ (1 << (73 - fb))
    sy = compute_syndrome(om, nc)
    s = l2i(sy)
    if bf != fb:
        check_error(s)
"""

# Then double bit errors
def check_double_error():
    print("Testing 100 random numbers with all Double bit errors")
    for _ in range(0, 100):
        cb = compute_checkbits(om, x)
        checkbits = l2i(cb)
        for i in range(0, k + r + 0):
            for j in range(i + 1, k + r + 1):
                bv = (x << (r + 1)) | checkbits
                bv  = bv ^ (1 << i)
                bv  = bv ^ (1 << j)
                sy = compute_syndrome(om, bv)
                s = l2i(sy[0:8])
                assert check_error(l2i(sy)) == 4,  "Double bit error uncorreclty classified as something else !"

"""
check_no_error()
check_single_bit_data_error()
check_single_bit_checkbits_error()
check_double_error()
"""
"""
# Same as all sec-ded codes, the number of the column that matches
# is the bit position to flip
print("Single bit error: syndrome contains erroneous bit column pattern")
for i in range(0, k):
    y = x ^ (0x1 << i)
    sdr = compute_syndrome(om, (y << (r + 1)) | checkbits)
    print(sdr, "⇒ ", end='')
    # Pattern matches erroneous bit
    w = np.where(np.all(mo == sdr, axis = 1))
    print(w[0])

print("Double bit error: checking procedure isn't that easy, I am afraid!")
for i in range(0, k):
    y = x ^ (0x3 << i)
    sdr = compute_syndrome(om, (y << (r + 1)) | checkbits)
    print(sdr, parity(l2i(sdr)))
"""

sys.exit(0)
