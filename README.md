# Two SEC-DED Codes in python

Python implentation of Hsiao and Lala secded schemes to protect 64-bit data.

There are 216 ones in Hsiao PCM and 202 in Lala's, but Lala's decoding needs a modulo 3 operation on the 8 syndrome bits.
We'll see when it comes to hardware implementation if this is acceptable.


Taken from Hsiao 1970 paper, using Figure 6 as parity check matrix, for Hsiao codes

* M.-Y. H SIAO. “A class of optimal minimum odd-weight-column SEC-DED codes”. In : IBM Journal of Research and Development 14.4 (1970), p. 395-401.

Taken from Lala's set of papers from 2003 to 2007, with some hacking due to not being able to correctly characterize the different types of error when using the procedure described in the paper.

* P. K. LALA. “A single error correcting and double error detecting coding scheme for computer memory systems”. In : Proceedings 18th IEEE Symposium on Defect and Fault Tolerance in VLSI Systems. IEEE. 2003, p. 235-241.
* P. LALA, P T HENAPPAN et M. A NWAR. “Single error correcting and double error detecting coding sche- me”. In : Electronics Letters 41.13 (2005), p. 758-760.
* M. A NWAR, P. LALA et P T HENAPPAN. “Decoder design for a new single error correcting/double error detecting code”. In : Proceeding of world academy of science 22.4 (2007), p. 247.

Note that Lala's work seems to have gone under the radar, for a reason I can't really explain (although the error checking procedure given in the 3 papers differs, and none of them is not the one I ended up implementing).
