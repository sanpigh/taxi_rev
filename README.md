# Data analysis
- Document here the project: taxi_rev
- Description: Librery for data regression.
- Type of regression: linear regression (lasso, ridge, linear), random tree, DecisionTree, Xgboost

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Functionnal test with a script:

```bash
scripts/taxi_rev-run
```

# Install

Go to `https://github.com/{group}/taxi_rev` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/taxi_rev.git
cd taxi_rev
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
taxi_rev-run
```
