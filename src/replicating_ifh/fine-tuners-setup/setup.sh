#!/usr/bin/env bash

function setup_local() {
    bash "fine-tuners-setup/$1.sh" \
    > "$1.log" 2>&1
}

function setup_remote() {
    wget -qO- "https://raw.githubusercontent.com/cochaviz/mep/experiments/src/replicating_ifh/fine-tuners-setup/$1.sh" \
    | bash \
    > "$1.log" 2>&1
}

remote=false

# check if user is running the script with the correct number of arguments
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <method> [--remote]"
    exit 1
fi

# check if the user wants to set up all fine-tuners
if [[ $1 == "all" ]]; then
    methods=("lmbff" "adapet")
else
    methods=($1)
fi

# check if the user wants to set up the fine-tuners from a remote repository
if [[ $2 == "--remote" ]]; then
    remote=true
    echo "Running from remote repository"
else
    echo "Running from local repository"

    # check if user is in the correct directory
    if [[ ! -d "fine-tuners-setup" ]]; then
        echo "Please run this script one directory above fine-tuners-setup."
        exit 1
    fi
fi

# set up the fine-tuners
for method in "${methods[@]}"; do
    echo "Setting up: $method..."

    if [[ $remote -eq true ]]; then
        setup_remote $method
    else
        setup_local $method
    fi

    if [[ $? -eq 0 ]]; then
        echo "Successfully set up: $method!"
    else
        echo "Failed setting up: $method... Check logs."
    fi
done