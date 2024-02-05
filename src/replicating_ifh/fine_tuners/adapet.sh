# #!/usr/bin/bash

dirname="fine_tuners/adapet"

if [ -d "$dirname" ]; then
    echo "Directory $dirname already exists. Please remove it in case you'd like to reset the setup."
    exit 2
fi

git clone git@github.com:cochaviz/ADAPET.git $dirname \
    || git clone https://github.com/cochaviz/ADAPET.git $dirname

cd $dirname && git pull 

# BEGIN rrmenon10/ADAPET/blob/master/bin/init.sh
if [ ! -d "data/superglue/" ] ; then
    mkdir -p data/superglue
    cd data/superglue

    wget "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip"
    unzip combined.zip

    cd -
fi

if [ ! -d "data/fewglue/" ] ; then
    git clone https://github.com/timoschick/fewglue.git data/fewglue
    cd data/fewglue

    rm -rf .git
    rm README.md
    mv FewGLUE/* .
    rm -r FewGLUE

    cd -
fi
# END rrmenon10/ADAPET/blob/master/bin/init.sh

# create the conda environment if it doesn't exist
conda env list | grep "adapet" \
    && echo "Conda environment adapet already exists" \
    || conda create -n adapet -y python=3.8

# install dependencies with conda
conda run -n adapet pip install -r requirements.txt

# set environment variables
conda env config vars set ADAPET_ROOT="$PWD" -n adapet
conda env config vars set PYTHONPATH="$PWD:$PYTHONPATH" -n adapet
