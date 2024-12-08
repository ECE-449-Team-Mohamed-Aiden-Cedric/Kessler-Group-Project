# Kessler-Group-Project

## How to Run

Our program has been tested to work on Python 3.12-3.13, and may not work on older versions.

First, install necessary requirements with `pip install -r requirements.txt`.

The model can be trained, or the best result from a saved model can be run.

The config file contains parameters that can be easily modified.
Important parameters to note for running the best result are:

1. `RUN_WITH_GRAPHICS`
2. `GA_MODEL_FILE` (if it was changed)

The rest of the parameters are mainly used for learning and should not be changed.

`run_best_result.py` is used to, as the name suggests, run the best result.
This loads the best result from the `GA_MODEL_FILE` using `pygad`, and then runs
our controller with the best chromosome in that saved state. This means that you
must have `pygad` installed (it is in `requirements.txt` after all) to run our controller.

If our controller must be run in a different way, other files in the `src` directory
must still be included anyway as the controller may use them for type hints or various
other things. In theory our controller can be run without `pygad` by manually entering
the `list` of `float`s best chromosome that is stored in the results in the `GA_MODEL_FILE`.

To train the model, run `genetic_learner.py`, making sure to first set flags appropriately
in the config file , such as disabling graphics and setting a reasonable number of
processes for your hardware.

If you have any issues, please email me at <cjbouche@ualberta.ca>.

## Team Members

- Cedric Boucher
- Aiden Teal
- Mohamed Alzinnad
