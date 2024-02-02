methods=("lmbff" "adapet")

if [[ $1 -eq "--remote" ]]; then
    echo "Running from remote repository"

    for method in "${methods[@]}"; do
        echo "Running $method"
        wget -qO- "https://raw.githubusercontent.com/cochaviz/mep/experiments/src/replicating_ifh/fine-tuners-setup/$method.sh" | bash | tee "$method.log"

        if [[ $? -eq 0 ]]; then
            echo "Successfully set up: $method!"
        else
            echo "Failed setting up: $method... Check logs."
        fi
    done
else
    echo "Running from local repository"

    # check whether user is running the script from withing the folder
    # 'fine-tuners-setup'
    if ! [[ -f "all.sh" ]]; then
        echo "Please run the script from within the folder 'fine-tuners-setup'"
        exit 1
    fi

    cd ..

    for method in "${methods[@]}"; do
        echo "Running $method"
        bash "fine-tuners-setup/$method.sh" | tee "fine-tuners-setup/$method.log"

        if [[ $? -eq 0 ]]; then
            echo "Successfully set up: $method!"
        else
            echo "Failed setting up: $method... Check logs."
        fi
    done

    cd -
fi
```