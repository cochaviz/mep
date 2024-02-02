# #!/usr/bin/bash

# clone the repo if it doesn't exist
if ! [ -f "LM_BFF" ]; then
    git clone git@github.com:princeton-nlp/LM-BFF.git LM_BFF \
        || git clone https://github.com/princeton-nlp/LM-BFF.git LM_BFF
fi

cd LM_BFF && git pull 

# # create the conda environment if it doesn't exist
conda env list | grep "lmbff" \
    && echo "Conda environment lmbff already exists" \
    || conda create -n lmbff -y python=3.6

# to fix a package conflict caused by the regex package, 
# we just remove it from the requirements.txt file
echo -e "$(cat requirements.txt \
    | grep -v 'regex' \
    | grep -v 'certifi' \
)" > requirements-min.txt
conda run -n lmbff pip install -r requirements-min.txt \
    --ignore-installed certifi # google colab has certifi has a distutils package
                              # for _whatever reason_.

# download the dataset 
cd data
if ! [ -f "original" ]; then
    rm datasets.tar && echo "Removed old 'datasets.tar'"
    chmod +x download_dataset.sh
    ./download_dataset.sh
fi
cd ..

# generate the k-shot data
conda run -n lmbff python tools/generate_k_shot_data.py