# k8s_nnUNet

## Create Repository on GitHub

* Create Repository on GitHub with readme.md, LICENSE and .gitignore (template from the specific project, i.e python)
* Link Repository to deep-modular project
* Block direct push and commit on main branch on **Settings | Branches** and check **Require pull request reviews before
  merging**, **Dismiss stale pull request approvals when new commits are pushed** and **Include administrators**

## Open GitHub Repository as PyCharm Project

* **Get from VCS**, select GitHub Repository and clone it in the desired path

## Configure PyCharm Project

### SSH Configuration

* **File | Settings | Tools | SSH Configurations**

### Deployment Configuration

* **File | Settings | Build, Execution, Deployment | Deployment**

To deploy on remote server:

* **Tools | Deployment | Upload To**

Deploy `.idea` and `.git` folders on remote, selecting also **Delete remote files when locals are deleted** in
**Tools | Deployment | Options**

### Docker Configuration

* **View | Tool Windows | Services | Add service | Docker**

### Kubernetes Configuration

* **View | Tool Windows | Services | Add service | Kubernetes**

## Create Issue

On GitHub

* **Issues | New issue**

Set Title and Description, assign Issue, link Project and set Labels

## Work on Issue

On PyCharm

* **Tools | Tasks & Contexts | Open Task**

Associate changelist and create new Git branch, update Issue state as open.

The GitHub server account can be linked at :

* **File | Settings | Tools | Tasks | Servers**

## Git

New unversioned files added on Git or to .gitignore on **Right Click | Git | Add or Add to .gitignore**

To remove from VCS a versioned file : `git rm --cached FILEPATH` and commit

## Commit and Push

Commit changes in the specific Changelist by selecting the files in **View | Tool Windows | Commit** and clicking
**Commit** or **Commit and Push**

Pull commits from origin with **Git | Update Project...**
## Create Pull Request

On **Git | GitHub | Create Pull Request...** create the request, filling the fields for the Reviewers, Assigned developer, Labels, Title and optional Description.

## Merge Branch

## Close Task

## Setup pre-commit and MyPy

[Pre-commit](https://pre-commit.com)
* Generate a sample config with `pre-commit sample-config > .pre-commit-config.yaml`
* Set the repos, and their corresponding hooks in `.pre-commit-config.yaml`
* Run `pre-commit install`
* `pre-commit run`

*MyPy* can be found in the conda environment

Run Mypy with `--config-file` [tox.ini](tox.ini), as described in [Tox](#tox)

To properly run Black hook, [pyproject.toml](pyproject.toml) should be present in the project folder

## Remote Python to Run and Debug code

Create SSH Python Interpreter, linking to the corresponding *python* (either *system* or *anaconda* interpreter).

Link the Project Folders, disabling the Automatic upload.

## Check Requirements

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

To run Tox:

* `tox -e LIST_OF_VIRTUAL_ENVS`

## Package repository

Create *tar.gz* package:

* `python setup.py sdist`

Create *whl* package:

* `python setup.py bdist_wheel`

