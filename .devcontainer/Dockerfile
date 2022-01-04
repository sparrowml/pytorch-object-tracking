FROM python:3.9

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" && echo $SNIPPET >> "/root/.bashrc"

RUN pip install poetry==1.1.6

WORKDIR /code
COPY poetry.lock poetry.toml pyproject.toml /code/
RUN poetry install
