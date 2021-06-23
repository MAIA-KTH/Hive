# deep-modular_nnUNet

## Create Repository on GitHub

* Create Repository on GitHub with readme.md, LICENSE and .gitignore (template from the specific project, i.e python)
* Link Repository to deep-modular project
* Block direct push and commit on main branch on **Settings | Branches** and check **Require pull request reviews before
  merging**, **Dismiss stale pull request approvals when new commits are pushed** and **Include administrators**

## Open GitHub Repository as PyCharm Project

* **Get from VCS**, select GitHub Repository and clone it in the desired path

## Configure PyCharm Project

### SSH Configuration

* **File | Settings-Tools- | SSH Configurations**

### Deployment Configuration

* **File | Settings | Build, Execution, Deployment | Deployment**

To deploy on remote server:

* **Tools | Deployment | Upload To**

Deploy `.idea` and `.git` folders on remote, selecting also **Delete remote files when locals are deleted** in
**Tools | Deployment | Options**

### Docker Configuration

* **View | Tool Windows | Services | Add service | Docker**

## Create Issue

On GitHub

* **Issues | New issue**

Set Title and Description, assign Issue, link Project and set Labels

## Work on Issue

On PyCharm

* **Tools | Tasks & Contexts | Open Task**

Associate changelist and create new Git branch, update Issue state as open.

## Git

New unversioned files added on Git or to .gitignore on **Right Click | Git | Add or Add to .gitignore**

To remove from VCS a versioned file : `git rm --cached FILEPATH` and commit

## Commit and Push

## Create Pull Request

## Merge Branch

## Close Task

## Setup Conda environment inside container

* `source /opt/conda/bin/activate`
* `conda create -n nnUNet --clone base`
* `conda activate nnUNet`
* `conda install -c conda-forge --file `[conda_requirements.txt](conda_requirements.txt)

## Setup pre-commit and MyPy

[Pre-commit](https://pre-commit.com)
* Generate a sample config with `pre-commit sample-config > .pre-commit-config.yaml`
* Set the repos, and their corresponding hooks in `.pre-commit-config.yaml`
* Run `pre-commit install`
* `pre-commit run`

*MyPy* can be found in the conda environment

Run Mypy with `--config-file` [tox.ini](k8s_data_preparation/tox.ini), as described in [Tox](#tox)

To properly run Black hook, [pyproject.toml](pyproject.toml) should be present in the project folder

## Check Requirement

Install pipreqs: `pip install pipreqs`

Run `pipreqs /PATH/TO/FOLDER`

## Testing code

Pytest is used to test python code. Every function starting with *test_*, in a file starting with *test_*, will be
tested.

To test python code, run : `pytest /path/to/test`

## Tox

Tox is used to test and package the python project.

`tox-quickstart` create an initial tox config file, named *tox.ini*.

*Mypy*, *Pytest* and *Flake8* configurations are added in *tox.ini*.

## Package repository

create_package setuptools