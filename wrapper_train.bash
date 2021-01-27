if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

main_path="$PROJECT_PATH"/main.py

if [ "$PYTHONPATH" = "" ]; then
  export PYTHONPATH=${PYTHONPATH}:"$PROJECT_PATH"/
fi

source "$PROJECT_PATH"/venv/bin/activate

python "$main_path" train