if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

main_path="$PROJECT_PATH"/main.py

if [ "$PYTHONPATH" = "" ]; then
  export PYTHONPATH=${PYTHONPATH}:"$PROJECT_PATH"/
fi

source "$PROJECT_PATH"/venv/bin/activate

# todo probably add some control here if exists a file in saved_generator directory
python "$main_path" generation