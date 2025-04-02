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
from contextlib import redirect_stdout

np.set_printoptions(threshold=sys.maxsize)

k = 64   # Number of bits to protect
r = 8    # Number of check bits according to theory

# Dump something looking like Verilog
def dump_verilog(pcm):
    # Looks like we need to have several independent files
    with open('compute_checkbits_hsiao.v', 'w') as f:
        with redirect_stdout(f):
            print("module compute_checkbits_hsiao (\n",
                  "    input  wire [63:0] d,\n",
                  "    output wire  [7:0] c\n);\n")

            for c in range(0, r):
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

    with open('compute_syndrome_hsiao.v', 'w') as f:
        with redirect_stdout(f):
            print("module compute_syndrome_hsiao (\n",
                  "    input  wire [63:0] d,\n",
                  "    input  wire  [7:0] c,\n",
                  "    output wire  [7:0] s\n);\n")

            for s in range(0, r):
                print(f"assign s[{s}] = ", end='')
                start = 0;
                for d in range(0, k + r):
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

    with open('check_syndrome_hsiao.v', 'w') as f:
        with redirect_stdout(f):
            print("module check_syndrome_hsiao (\n",
                  "    input  wire  [7:0] syndrome,\n",
                  "    output wire        ne, // no error\n",
                  "    output wire        se, // single bit error\n",
                  "    output wire        de  // double bit error\n",
                  ");\n")

            print(f"assign ne = syndrome == 0;")
            print(f"assign se = syndrome[7] ^ syndrome[6] ^ syndrome[5] ^ syndrome[4] ^ syndrome[3] ^ syndrome[2] ^syndrome[1] ^ syndrome[0];")
            print(f"assign de = (syndrome == 0) && !se;")
            print("endmodule")


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
dump_verilog(om)
