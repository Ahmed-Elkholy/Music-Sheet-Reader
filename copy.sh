#!/bin/bash
m=0
for file in $1/*/*; do
    eval "cp $file $2/$m.jpg";
    m=$[ $m + 1 ];
done
