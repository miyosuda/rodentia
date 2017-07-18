#!/bin/sh

if [ -e src/python/rodent_module.so ]; then
rm src/python/rodent_module.so
fi

cp src/librodent_module.dylib src/python/rodent_module.so

python src/python/rodent_test.py
