# #!/usr/bin/bash

# clone the repo if it doesn't exist
if ! [ -d "ADAPET" ]; then
    git clone git@github.com:rrmenon10/ADAPET.git ADAPET \
        || git clone https://github.com/rrmenon10/ADAPET.git ADAPET
fi

cd ADAPET && git pull 

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