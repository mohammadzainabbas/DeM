## Setup `pre-commit` hooks 👨🏻‍💻

### Table of contents

- [Setup `pre-commit` hooks](#setup-pre-commit)

#

<a id="setup-pre-commit" />

#### 1. Setup `pre-commit` hooks

Git hook scripts are useful for identifying simple issues before submission to code review. We run our hooks on every commit to automatically point out issues in code such as _missing semicolons_, _trailing whitespace_, and _debug statements_. Checkout `pre-commit` hook [site](https://pre-commit.com/index.html) for more details.

You can setup `pre-commit` hook by running:

```bash
brew install pre-commit
```

or if you prefer `pip`

```bash
pip install pre-commit
```

You can have a look at some basic pre-commit hooks that I have added under [pre-commit-config.yml](https://github.com/mohammadzainabbas/MGMA-Lab/blob/lab1/pre-commit-config.yml) file. You can checkout all the `pre-commit hooks` [here](https://pre-commit.com/hooks.html). And add any hook that you like in this file.

Now, install the pre-commit hook by running:

```bash
pre-commit install
```

you will see

```txt
pre-commit installed at .git/hooks/pre-commit
```

This means pre-commit hook were installed successfully.
