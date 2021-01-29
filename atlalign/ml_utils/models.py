"""Module containing utilities for manipulation of keras models."""

"""
    The package atlalign is a tool for registration of 2D images.

    Copyright (C) 2021 EPFL/Blue Brain Project

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import json
import pathlib
from copy import deepcopy

from keras.layers import Lambda, concatenate
from keras.models import Model
from keras.models import load_model as load_model_keras
from keras.models import model_from_json

from atlalign.ml_utils import (
    Affine2DVF,
    BilinearInterpolation,
    DVFComposition,
    ExtractMoving,
    NoOp,
)


def merge_global_local(model_g, model_l, expose_global=False):
    """Merge a global and local aligner models into a new one.

    Seems to be also changing the input models.

    Parameters
    ----------
    model_g : keras.Model
        Model performing global alignment.

    model_l : keras.Model
        Model performing local alignment.

    expose_global : bool, optional
        If True, then model has 4 outputs where the last two represent image after global alignment
        and the corresponding dvfs. If False, then only 2 outputs : image after both global and
        local and the overall dvfs.

    Returns
    -------
    model_gl : keras.Model
        Model performing both the local and the global alignment.
    """
    # define slicing layers
    extract_0 = Lambda(lambda x: x[..., :-1])

    is_inverse = isinstance(model_l.inputs, list) and len(model_l.inputs) == 2
    # prepare some tensors
    overall_input = model_g.input
    img_ref = extract_0(overall_input)

    img_reg_g, dvfs_g = model_g.outputs[
        :2
    ]  # in case it has 3 outputs i.e inverse model
    middle_input = concatenate([img_ref, img_reg_g])

    if is_inverse:
        middle_input = [middle_input, middle_input]  # quick hack

    img_reg_gl, dvfs_l = model_l(middle_input)[:2]

    dvfs_gl = DVFComposition()([dvfs_g, dvfs_l])

    new_output = [img_reg_gl, dvfs_gl]

    if expose_global:
        new_output += [img_reg_g, dvfs_g]

    model_gl = Model(inputs=overall_input, outputs=new_output)

    return model_gl


def save_model(model, path, separate=True, overwrite=True):
    """Save model.

    Parameters
    ----------
    model : keras.Model
        Keras model to be saved.

    path : str or pathlib.Path
        Path to where to save the serialized model. If `separate=True` then it needs to represent a folder name.
        Inside of the folder 2 files are created - weights (.h5) and architecture (.json). If `separate=False` then
        an extension `.h5` is added and architecture + weights are dumped into one file.

    separate : bool
        If True, then architecture and weights are stored separately. Note that if False then one might encounter
        issues when loading in in a different Python environment (see references).

    overwrite : bool
        If True, then possible existing files/folders are overwritten.

    References
    ----------
    [1] https://github.com/keras-team/keras/issues/9595

    """
    path = pathlib.Path(path)

    if path.suffix:
        raise ValueError("Please specify a path without extension (folder).")

    if not separate:
        model.save(str(path) + ".h5", overwrite=overwrite)

    else:
        path_architecture = path / (path.stem + ".json")
        path_weights = path / (path.stem + ".h5")

        if path_architecture.exists() and not overwrite:
            raise FileExistsError(
                "The file {} already exists and overwriting is disabled.".format(
                    path_architecture
                )
            )

        if not path_architecture.exists():
            path_architecture.parent.mkdir(
                parents=True, exist_ok=True
            )  # maybe the folder was already created before
            path_architecture.touch()

        with path_architecture.open("w") as f_a:
            json.dump(model.to_json(), f_a)

        model.save_weights(str(path_weights), overwrite=overwrite)


def load_model(path, compile=False):
    """Load a model that uses custom `atlalign` layers.

    The benefit of using this function as opposed to the keras equivalent is that the user does not have to care about
    how the model was saved (whether architecture and weights were separated). Additionally, all custom possible
    custom layers are provided.

    Parameters
    ----------
    path : str or pathlib.Path
        If `path` is a folder then the folder is expected to have one `.h5` file (weights) and one `.json`
        (architecture). If `path` is a file then it needs to be an `.h5` and it needs to encapsulate both
        the weights and the architecture.

    compile : bool
        Only possible if `path` refers to a `.h5` file.

    Returns
    -------
    keras.Model
        Model ready for inference. If `compile=True` then also ready for continuing the training.

    """
    path = pathlib.Path(str(path))

    if path.is_file():
        model = load_model_keras(
            str(path),
            compile=compile,
            custom_objects={
                "Affine2DVF": Affine2DVF,
                "DVFComposition": DVFComposition,
                "BilinearInterpolation": BilinearInterpolation,
                "ExtractMoving": ExtractMoving,
                "NoOp": NoOp,
            },
        )

    elif path.is_dir():
        if compile:
            raise ValueError(
                "Cannot compile the model because weights and architecture stored separately."
            )

        h5_files = [p for p in path.iterdir() if p.suffix == ".h5"]
        json_files = [p for p in path.iterdir() if p.suffix == ".json"]

        if not (len(h5_files) == 1 and len(json_files) == 1):
            raise ValueError(
                "The folder {} needs to contain exactly one .h5 file and one .json file".format(
                    path
                )
            )

        path_architecture = json_files[0]
        path_weights = h5_files[0]

        with path_architecture.open("r") as f:
            json_str = json.load(f)

        model = model_from_json(
            json_str,
            custom_objects={
                "Affine2DVF": Affine2DVF,
                "DVFComposition": DVFComposition,
                "BilinearInterpolation": BilinearInterpolation,
                "ExtractMoving": ExtractMoving,
                "NoOp": NoOp,
            },
        )
        model.load_weights(str(path_weights))

    else:
        raise OSError("The path {} does not exist.".format(str(path)))

    return model


def replace_lambda_in_config(input_config, output_format="dict", verbose=False):
    """Replace Lambda layers with full blown keras layers.

    This function only exists because we pretrained a lot of models with 2 different
    Lambda layers and only after that realized that they cause issues during
    serialization. One can then use this function to just fix it.

    Notes
    -----
    To make this clear let us define the top dictionary as the one that has keys

        - 'backend'
        - 'config'
        - 'keras_version'
        - 'class_name'

    The bottom dictionary is top['config'].

    Parameters
    ----------
    input_config : dict or str or pathlib.Path or keras.Model
        Config containing an architecture generated by one of the functions in `atlalign.nn` possibly
        containing Lambda layers.

    output_format : str, {'dict', 'keras', 'json'}
        What output type to use. See how below how to instantiate a model out of each of the formats

        - 'dict' - `keras.Model.from_config`
        - 'json' - `kersa.models.model_from_json`
        - 'keras' - already a model instance

    verbose : bool
        If True, printing to standard output.

    Returns
    -------
    output_config : dict
        Config containing an architecture of the input network but all Lambda layers are replaced by full
        blown operations.

    """
    translation_map = {"extract_moving": "ExtractMoving", "inv_dvf": "NoOp"}

    def lambda_replacer(layer_dict, verbose=False):
        """Replace a layer specific dict (only if Lambda layer).

        Parameters
        ----------
        layer_dict : dict
            Corresponds to an element of `json.loads(model.to_json())['config']['layers']`.

        verbose : bool
            If True, printing to standard output.

        Returns
        -------
        If not a Lambda layer then returns an unmodified `layer_dict`. However if it is a Lambda layer
        it is replaced by a predefined custom (non-Lambda) Layer.

        """
        copy_dict = deepcopy(layer_dict)
        if "class_name" not in copy_dict:
            raise KeyError("Does not contain class_name")

        if copy_dict["class_name"] != "Lambda":
            if verbose:
                print("Not a Lambda")
            return copy_dict

        if "config" not in copy_dict:
            raise KeyError("Does not contain config")

        name = copy_dict["config"]["name"]

        if name in translation_map:
            copy_dict["class_name"] = translation_map[name]
            new_config = {
                "name": name,
                "trainable": True,
            }
            copy_dict["config"] = new_config

        else:
            raise KeyError(
                "Stumbled upon a lambda layer with an unrecognized name: {}".format(
                    name
                )
            )

        return copy_dict

    if isinstance(input_config, str):
        # assuming it came from model.to_json()
        config_dict = json.loads(input_config)["config"]

    elif isinstance(input_config, Model):
        config_dict = input_config.get_config()

    elif isinstance(input_config, dict):
        config_dict = input_config

    elif isinstance(input_config, pathlib.Path):
        if input_config.suffix != ".json":
            raise ValueError(
                "The only allowed extension is .json, {} is unsupported".format(
                    input_config.suffix
                )
            )

        with input_config.open("r") as f:
            config_dict = json.loads(json.load(f))["config"]
    else:
        raise TypeError(
            "Unsupported type of input_config: {}".format(type(input_config))
        )

    hacked_config = deepcopy(config_dict)
    hacked_config["layers"] = []

    for i, x in enumerate(config_dict["layers"]):
        hacked_config["layers"].append(lambda_replacer(x, verbose=verbose))
        if verbose:
            print("Before == after: {}".format(x == hacked_config["layers"][-1]))
            print("\n\n")

    if output_format == "json":
        return Model.from_config(
            hacked_config,
            custom_objects={
                "Affine2DVF": Affine2DVF,
                "DVFComposition": DVFComposition,
                "BilinearInterpolation": BilinearInterpolation,
                "ExtractMoving": ExtractMoving,
                "NoOp": NoOp,
            },
        ).to_json()

    elif output_format == "dict":
        return hacked_config

    elif output_format == "keras":
        final_model = Model.from_config(
            hacked_config,
            custom_objects={
                "Affine2DVF": Affine2DVF,
                "DVFComposition": DVFComposition,
                "BilinearInterpolation": BilinearInterpolation,
                "ExtractMoving": ExtractMoving,
                "NoOp": NoOp,
            },
        )

        if isinstance(input_config, Model):
            final_model.set_weights(input_config.get_weights())

        return final_model

    else:
        raise TypeError("Unrecognized output format: {}".format(output_format))
