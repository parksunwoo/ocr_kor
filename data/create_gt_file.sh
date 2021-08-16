#!/bin/bash

echo $1

searchdir=./generator/TextRecognitionDataGenerator/out/$1

for entry in $searchdir/*

do
        filedir="${entry:2}"
        filename="${filedir##*/}"
        label="${filename%_*}"
        echo -e "$filedir\t$label"

done