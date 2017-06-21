#!/bin/sh

if [ -e src/python/rodent.so ]; then
rm src/python/rodent.so
fi

cp src/libpyrodent.dylib src/python/rodent.so

python src/python/rodent_test.py



