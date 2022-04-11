#!/bin/bash

for i in $(seq 1 1 100); do
    echo "Running trial $i out of 100"
    python3 -m unittest >tmp 2>&1
    if [ $? -ne 0 ]; then
        cat tmp
        rm -f tmp
        exit 1
    fi
done
rm -f tmp

echo ""
echo "All tests passed!"

