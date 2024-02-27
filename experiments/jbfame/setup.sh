#!/usr/bin/env bash

git clone https://github.com/verazuo/jailbreak_llms.git data

if [ $? -ne 0 ]; then
    exit 1;
fi

cd data

rm README.md
rm LICENSE
rm -rf .git

mv data/*.csv .
rm -rf data

cd -