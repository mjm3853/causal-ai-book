# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Code and exercises for the "Causal AI" book by Robert Ness.

- Book: https://www.manning.com/books/causal-ai
- Resources: https://www.robertosazuwaness.com/workshops.html

## Development Commands

```bash
# Install dependencies
uv sync

# Run Jupyter notebooks
uv run jupyter lab

# Run tests
uv run pytest
```

## Key Dependencies

- **pgmpy**: Probabilistic graphical models - used for Bayesian networks, discrete probability distributions, and causal inference
- **pyro-ppl**: Probabilistic programming on PyTorch - used for continuous distributions, sampling, and inference
- **torch**: Backend for Pyro
- **matplotlib**: Visualization

## Architecture

The repository follows the book's chapter structure:
- `ch2/` - Probability primer: marginal/joint/conditional distributions, Bayes' rule, sampling, Monte Carlo
- `ch3/` - Causal graphs: DAGs, Bayesian networks, fitting models to data

Each chapter contains Jupyter notebooks with documented examples and exercises.

## Common Patterns

### Discrete Distributions with pgmpy

```python
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Define discrete factors
dist = DiscreteFactor(variables=["X"], cardinality=[3], values=[0.45, 0.30, 0.25])

# Build Bayesian networks
model = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])
model.add_cpds(cpd_a, cpd_b, cpd_c)
model.fit(data)  # Learn CPDs from pandas DataFrame
```

### Continuous Distributions with Pyro

```python
import pyro
from pyro.distributions import Gamma, Poisson, Bernoulli

def model():
    z = pyro.sample("z", Gamma(7.5, 1.0))
    x = pyro.sample("x", Poisson(z))
    with pyro.plate("IID", 10):  # Vectorized IID sampling
        y = pyro.sample("y", Bernoulli(x / (5 + x)))
    return y
```

### Data Sources

The book uses datasets from: `https://raw.githubusercontent.com/altdeep/causalML/master/datasets/`
