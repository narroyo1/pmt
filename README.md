# Pin Movement Training
# A novel method for training generative machine learning models

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

## Table of Contents
- [Usage](#usage)
- [Documentation](#documentation)
- [License](#license)

## Usage
### Using VS Code
Open `main.py` and run.

### Tensorboard visualization
On the same directory as `main.py` run:
```shell
tensorboard --logdir=runs
```

Then open `http://localhost:6006/` on any browser.

The default experiment is the second one shown in the article, to reproduce any other change the following line on main.py
```
experiment = experiments.EXPERIMENT_<experiment name>
```

## Documentation

A paper throughly explaining the method is located at [Approximating Stochastic Functions with Multivariate Outputs](https://towardsdatascience.com/approximating-stochastic-functions-with-multivariate-outputs-ffefc7099a90).

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
