black slam_datagen/

isort slam_datagen/

printf "\nPress any key to continue to pylint...\n"
read -n 1 -s -r
pylint slam_datagen/

printf "\nPress any key to continue to mypy...\n"
read -n 1 -s -r
mypy slam_datagen/
