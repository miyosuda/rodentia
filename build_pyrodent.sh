#!/bin/sh

cd src

if [ -e pyrodent.so ]; then
rm pyrodent.so
fi

ln -s libpyrodent.dylib pyrodent.so

python check.py
