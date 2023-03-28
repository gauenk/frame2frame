#!/bin/bash
# Computes tvl1 optical flow for a (noisy) sequence.

I=${1:-""}
F=${2:-1}
L=${3:-1}
O=${4:-""}

for i in `seq $F $L`;
do
    ./tvl1flow `printf $I $((i+1))` \
        `printf $I $i` \
        `printf $O"_bflow.flo" $((i+1))` \
        4 0.25 0.2 0.3 100 2 0.5 5 0.01 0;
    ./tvl1flow `printf $I $i` \
        `printf $I $((i+1))` \
        `printf $O"_fflow.flo" $i` \
        4 0.25 0.2 0.3 100 2 0.5 5 0.01 0;
done
