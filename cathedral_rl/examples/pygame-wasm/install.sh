#!/bin/bash
mkdir -p modules
cd modules

python -m pip download gymnasium==0.27.1 pettingzoo==1.22.3 numpy==1.22.4 git+https://github.com/elliottower/cathedral-rl.git

unzip -o '*.whl'
rm *.whl