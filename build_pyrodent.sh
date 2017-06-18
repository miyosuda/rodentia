#!/bin/sh

cd src

if [ -e rodent.so ]; then
rm rodent.so
fi

ln -s libpyrodent.dylib rodent.so

python check.py
