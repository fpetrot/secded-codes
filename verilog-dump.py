#!/usr/bin/env python3.12
import sys
from sys import exit
import random
import itertools as it
from more_itertools import distinct_permutations
import numpy as np
from contextlib import redirect_stdout

# Make a list of 0s and 1s an integer
np.set_printoptions(threshold=sys.maxsize)

# Constants! Don't think those can be changed
k = 64
r = 8

# Dump something looking like Verilog
# Following the lowrisc example, we concatenate the check bits
# on the MSB, as opposed to what the matrix says, but this
# should not change much
def dump_verilog(name, cmp, pcm):
    mpc = np.transpose(pcm)
    with open(f'generated/prim_secded_{name}_73_64_enc.sv', 'w') as f:
        with redirect_stdout(f):
            print(f"module prim_secded_{name}_73_64_enc (\n"
                  "    input  logic [63:0] in,\n"
                  "    output logic [72:0] out\n);\n"
                  "    always_comb begin : p_encode\n"
                  "        out[63:0] = in;" )
            for c in range(0, r + 1):
                print(f"        out[{72 - c}] = ^(in & 64'h{l2u(pcm[c][9:73]):016x});")
            print(f"    end\nendmodule : prim_secded_{name}_73_64_enc")

    with open(f'generated/prim_secded_{name}_73_64_dec.sv', 'w') as f:
        with redirect_stdout(f):
            print(f"module prim_secded_{name}_73_64_dec (\n"
                  "    input  logic [72:0] in,\n"
                  "    output logic  [8:0] syndrome_o,\n"
                  "    output logic  [1:0] err_o\n);\n")

            for s in range(0, r + 1):
                print(f"    assign syndrome_o[{8 - s}] = ^(in & 73'h{l2u(pcm[s][0:73]):019x});")
            print("    logic  ne = (syndrome_o == 0);")
            print("    logic  se = ^syndrome_o;")
            print("    assign err_o = {~(ne | se), se};")
            print(f"endmodule : prim_secded_{name}_73_64_dec")

    with open(f'generated/prim_secded_{name}_73_64_cor.sv', 'w') as f:
        with redirect_stdout(f):
            print(f"module prim_secded_{name}_73_64_cor (\n"
                  "    input  logic [72:0] d_i,\n"
                  "    output logic [72:0] d_o,\n"
                  "    output logic  [8:0] syndrome_o,\n"
                  "    output logic  [1:0] err_o\n);\n")

            for s in range(0, r + 1):
                print(f"    assign syndrome_o[{8 - s}] = ^(d_i & 73'h{l2u(pcm[s][0:73]):019x});")
            if cmp == 1:
                for d in range(0, k + r + 1):
                    print(f"    assign d_o[{d}] = (syndrome_o == 9'h{l2i(pcm[:,72 - d]):03x}) ^ d_i[{d}];")
            else:
                for d in range(0, k):
                    print(f"    assign d_o[{d}] = (", end='')
                    n = 0
                    t = len(pcm[:,72 - d]) - 1
                    for i,b in enumerate(pcm[:,72 - d]):
                        if b:
                            n = n + 1
                            print(f"syndrome_o[{t - i}]", end='')
                            if n < 3:
                                print(" & ", end='')
                    print(f") ^ d_i[{d}];")
                for d in range(k, k + r + 1):
                    print(f"    assign d_o[{d}] = (syndrome_o == 9'h{l2i(pcm[:,72 - d]):03x}) ^ d_i[{d}];")

            print("    logic  ne = (syndrome_o == 0);")
            print("    logic  se = ^syndrome_o;")
            print("    assign err_o = {~(ne | se), se};")
            print(f"endmodule : prim_secded_{name}_73_64_cor")

# Make a list of 0s and 1s an integer
def l2i(l):
    v = 0
    t = len(l)
    for b in l:
        v = (v << 1) | b
    return int(v)

# Hack to output a 64-bit unsigned integer
def l2u(l):
    t = len(l)
    v = 1 << t
    for b in l:
        v = (v << 1) | b
    return (v - (1 << t)) & ((1 << t) - 1)


def l2i(l):
    v = 0
    t = len(l)
    for b in l:
        v = (v << 1) | b
    return int(v)

def binlist(pcm):
    intlist = []
    for i in range(37, 73):
        row = pcm[i,:]
        rl = row.tolist()
        x = l2i(rl)
        intlist.append(x)
    return intlist

def hamming_distance(code, edoc):
    xor = code ^ edoc
    return xor.bit_count()

def total_hamming_distance(pcm):
    thd = 0
    codes = binlist(pcm)
    n = len(codes)
    for i in range(n):
        for j in range(i + 1, n):
            thd = thd + hamming_distance(codes[i], codes[j])
    print(f"Total Hamming Distance : {thd}")

mat = np.zeros(shape=(9, 37), dtype=np.uint32)
#          c c c c c c c c c  d d d d d d d d  d d d d d d d d  d d d d d d d d  d d d d  #d d d d  d d d d d d d d  d d d d d d d d  d d d d d d d d  d d d d d d d d
#                             6 6 5 5 5 5 5 5  5 5 5 5 5 4 4 4  4 4 4 4 4 4 4 3  3 3 3 3  #3 3 3 3  3 3 2 2 2 2 2 2  2 2 2 2 1 1 1 1  1 1 1 1 1 1 0 0  0 0 0 0 0 0 0 0
#          8 7 6 5 4 3 2 1 0  3 2 1 9 8 7 6 5  4 3 2 1 0 0 9 8  7 6 5 4 3 2 1 0  9 8 7 6  #5 4 3 2  1 0 9 8 7 6 5 4  3 2 1 0 9 8 7 6  5 4 3 2 1 0 9 8  7 6 5 4 3 2 1 0
mat[0]  = [1,0,0,0,0,0,0,0,0, 1,0,0,0,1,1,0,1, 0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0, 0,0,1,0] #0,0,0,1, 0,0,1,0,1,0,1,0, 0,1,1,1,1,0,0,1, 0,1,1,1,0,0,0,0, 1,0,1,0,1,0,0,0] #23
mat[1]  = [0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1, 1,1,0,0,0,0,1,0, 1,0,1,0,1,0,0,0, 0,0,0,0] #0,0,0,1, 1,0,1,0,0,0,1,1, 1,0,1,0,0,0,0,0, 1,1,0,0,1,0,1,0, 0,0,0,0,1,1,1,0] #23
mat[2]  = [0,0,1,0,0,0,0,0,0, 0,0,0,1,0,1,0,0, 0,0,0,0,0,1,1,0, 0,0,0,0,0,0,1,1, 0,0,0,1] #0,1,0,0, 0,0,0,1,0,0,0,0, 0,0,0,0,0,0,0,1, 0,0,0,0,1,1,1,1, 0,1,1,1,1,0,1,0] #20
mat[3]  = [0,0,0,1,0,0,0,0,0, 1,0,0,0,0,0,0,0, 0,1,1,1,0,0,0,0, 0,0,0,0,0,1,0,1, 1,0,0,0] #0,1,0,0, 0,1,0,1,0,1,0,1, 1,1,0,1,1,1,0,1, 1,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,1] #22
mat[4]  = [0,0,0,0,1,0,0,0,0, 0,0,1,1,0,0,1,0, 0,0,0,0,0,0,0,1, 0,0,1,1,0,0,0,0, 1,0,0,0] #1,1,1,1, 1,0,0,0,1,1,0,0, 0,0,0,0,1,1,1,0, 1,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,0] #22
mat[5]  = [0,0,0,0,0,1,0,0,0, 0,1,0,0,0,0,1,0, 0,0,0,0,0,0,0,0, 1,1,0,0,0,1,0,0, 0,0,1,1] #0,0,1,0, 0,1,1,0,1,0,0,0, 0,1,0,0,0,1,1,0, 0,0,1,0,0,1,0,0, 0,0,0,0,0,0,1,1] #19
mat[6]  = [0,0,0,0,0,0,1,0,0, 0,0,1,0,0,0,0,0, 1,0,0,1,1,1,0,0, 0,1,0,0,0,0,0,0, 0,1,0,0] #1,0,0,0, 0,0,0,1,0,1,1,0, 1,0,0,1,0,0,1,0, 0,0,0,1,1,0,0,0, 1,1,0,1,0,1,0,1] #22
mat[7]  = [0,0,0,0,0,0,0,1,0, 0,1,0,0,1,0,0,0, 0,0,1,0,0,0,0,1, 0,0,0,0,1,0,1,0, 0,1,0,0] #1,0,1,0, 1,1,0,0,0,0,0,1, 0,0,1,0,0,0,0,0, 0,0,1,1,0,1,1,1, 0,0,1,1,0,1,0,0] #22
mat[8]  = [0,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1] #0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0] #29

lala = np.zeros(shape=(9, 73), dtype=np.uint32)
lala[0]  = [1,0,0,0,0,0,0,0,0, 1,0,0,0,1,1,0,1, 0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0, 0,0,1,0, 0,0,0,1, 0,0,1,0,1,0,1,0, 0,1,1,1,1,0,0,1, 0,1,1,1,0,0,0,0, 1,0,1,0,1,0,0,0] #23
lala[1]  = [0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1, 1,1,0,0,0,0,1,0, 1,0,1,0,1,0,0,0, 0,0,0,0, 0,0,0,1, 1,0,1,0,0,0,1,1, 1,0,1,0,0,0,0,0, 1,1,0,0,1,0,1,0, 0,0,0,0,1,1,1,0] #23
lala[2]  = [0,0,1,0,0,0,0,0,0, 0,0,0,1,0,1,0,0, 0,0,0,0,0,1,1,0, 0,0,0,0,0,0,1,1, 0,0,0,1, 0,1,0,0, 0,0,0,1,0,0,0,0, 0,0,0,0,0,0,0,1, 0,0,0,0,1,1,1,1, 0,1,1,1,1,0,1,0] #20
lala[3]  = [0,0,0,1,0,0,0,0,0, 1,0,0,0,0,0,0,0, 0,1,1,1,0,0,0,0, 0,0,0,0,0,1,0,1, 1,0,0,0, 0,1,0,0, 0,1,0,1,0,1,0,1, 1,1,0,1,1,1,0,1, 1,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,1] #22
lala[4]  = [0,0,0,0,1,0,0,0,0, 0,0,1,1,0,0,1,0, 0,0,0,0,0,0,0,1, 0,0,1,1,0,0,0,0, 1,0,0,0, 1,1,1,1, 1,0,0,0,1,1,0,0, 0,0,0,0,1,1,1,0, 1,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,0] #22
lala[5]  = [0,0,0,0,0,1,0,0,0, 0,1,0,0,0,0,1,0, 0,0,0,0,0,0,0,0, 1,1,0,0,0,1,0,0, 0,0,1,1, 0,0,1,0, 0,1,1,0,1,0,0,0, 0,1,0,0,0,1,1,0, 0,0,1,0,0,1,0,0, 0,0,0,0,0,0,1,1] #19
lala[6]  = [0,0,0,0,0,0,1,0,0, 0,0,1,0,0,0,0,0, 1,0,0,1,1,1,0,0, 0,1,0,0,0,0,0,0, 0,1,0,0, 1,0,0,0, 0,0,0,1,0,1,1,0, 1,0,0,1,0,0,1,0, 0,0,0,1,1,0,0,0, 1,1,0,1,0,1,0,1] #22
lala[7]  = [0,0,0,0,0,0,0,1,0, 0,1,0,0,1,0,0,0, 0,0,1,0,0,0,0,1, 0,0,0,0,1,0,1,0, 0,1,0,0, 1,0,1,0, 1,1,0,0,0,0,0,1, 0,0,1,0,0,0,0,0, 0,0,1,1,0,1,1,1, 0,0,1,1,0,1,0,0] #22
lala[8]  = [0,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1, 0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0] #29

ist = f'{"1" * 3}{"0" * 5}'
per = distinct_permutations(ist)
lpr = list(per)
lp = np.array(lpr)
lp = lp == '1'
lp = lp.astype(int)
z = np.zeros(shape=(56, 1), dtype=np.uint32)
lp = np.concatenate((lp, z), axis=1)

tam = np.concatenate((np.transpose(mat), lp), axis=0)

random.seed(0xdeadbeef)
for kk in range(100):
    lala = tam
    # Remove 10 rows randomly
    for i in range(20):
        rx = 37 + random.randrange(56 - i)
        lala = np.delete(lala, rx, axis=0)
    print("-------------------")
    total_hamming_distance(lala)
    print(sum(lala))
    print(sum(sum(lala)))
    dump_verilog(str(kk), 0, np.transpose(lala))

print(lala)
exit(1)
