"""Collection of tests focused on the atlalign.ml_utils.models module."""

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

import numpy as np
import pytest
from tensorflow import keras

from atlalign.ml_utils import (
    ExtractMoving,
    NoOp,
    load_model,
    merge_global_local,
    replace_lambda_in_config,
    save_model,
)


@pytest.fixture(scope="function", params=["compiled", "uncompiled"])
def reg_model(request):
    """Minimal registration network."""

    compile = request.param == "compiled"

    inputs = keras.layers.Input((32, 45, 2))

    extract_0 = keras.layers.Lambda(lambda x: x[..., :-1])
    dummy_op_1 = keras.layers.Lambda(lambda x: x + 1)
    dummy_op_2 = keras.layers.Lambda(lambda x: x * 1)

    reg_imgs = dummy_op_1(extract_0(inputs))
    dvfs = dummy_op_2(inputs)

    model = keras.models.Model(inputs=inputs, outputs=[reg_imgs, dvfs])

    if compile:
        model.compile(optimizer="adam", loss="mse")

    return model


@pytest.fixture()
def lambda_model():
    """Model containing Lambda layers"""
    inputs = keras.layers.Input(shape=(10, 14, 2))
    x = keras.layers.Lambda(lambda x: x, name="inv_dvf")(inputs)
    x = keras.layers.Lambda(lambda x: x[..., 1:], name="extract_moving")(x)

    return keras.Model(inputs=inputs, outputs=x)


class TestGlobalLocal:
    """Tests focused on the merge_global_local function."""

    @pytest.mark.parametrize("expose_global", [True, False])
    def test_overall(self, reg_model, expose_global):
        model_gl = merge_global_local(reg_model, reg_model, expose_global=expose_global)

        assert isinstance(model_gl, keras.models.Model)

        assert len(model_gl.outputs) == (4 if expose_global else 2)


class TestSaveModel:
    """Collection of tests focused on the `save_model` function."""

    @pytest.mark.parametrize("separate", [True, False])
    def test_correct(self, separate, tmpdir, reg_model):
        temp_path = pathlib.Path(str(tmpdir)) / "temp_model"
        save_model(reg_model, temp_path, separate=separate)

        if separate:
            assert (temp_path / "temp_model.json").exists()
            assert (temp_path / "temp_model.h5").exists()
        else:
            assert pathlib.Path(str(temp_path) + ".h5").exists()

    def test_wrong_path(self, reg_model):
        with pytest.raises(ValueError):
            save_model(reg_model, "aaa.extension")

    def test_already_exists(self, reg_model, tmpdir):
        orig_path = pathlib.Path(str(tmpdir)) / "temp_model"
        temp_path = orig_path / "temp_model.json"

        temp_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # maybe the folder was already created before
        temp_path.touch()

        with pytest.raises(FileExistsError):
            save_model(reg_model, orig_path, overwrite=False)


class TestLoadModel:
    """Collection of tests focused on the `load_model` function."""

    @pytest.mark.parametrize("separate", [True, False])
    @pytest.mark.parametrize("compile", [True, False])
    def test_correct(self, separate, compile, tmpdir, reg_model):
        original_compiled = reg_model.optimizer is not None

        temp_path = pathlib.Path(str(tmpdir)) / "temp_model"
        save_model(reg_model, temp_path, separate=separate)

        if separate and compile:
            with pytest.raises(ValueError):
                load_model(
                    str(temp_path) + "{}".format("" if separate else ".h5"),
                    compile=compile,
                )
            return

        model = load_model(
            str(temp_path) + "{}".format("" if separate else ".h5"), compile=compile
        )

        assert isinstance(model, keras.Model)

        if not separate and compile and original_compiled:
            assert model.optimizer is not None
        else:
            assert model.optimizer is None

        assert len(model.get_config()["layers"]) == len(
            reg_model.get_config()["layers"]
        )
        # assert reg_model.to_json() == model.to_json()  # might differ because of lambda layers:D:D

    def test_nonexistent_path(self, tmpdir):
        with pytest.raises(OSError):
            load_model(tmpdir / "fake")

    def test_ambiguous(self, tmpdir):
        path = pathlib.Path(str(tmpdir))
        path_architecture_1 = path / "a_1.json"
        path_architecture_2 = path / "a_2.json"
        path_weights = path / "w_1.h5"

        path_architecture_1.touch()
        path_architecture_2.touch()
        path_weights.touch()

        with pytest.raises(ValueError):
            load_model(path)


class TestReplaceLambdaInConfig:
    """Collection of tests focused on the `replace_lambda_in_config` method."""

    @pytest.mark.parametrize("input_format", ["json", "keras", "dict", "path"])
    @pytest.mark.parametrize("output_format", ["json", "keras", "dict"])
    def test_identical_results(self, lambda_model, input_format, output_format, tmpdir):

        if input_format == "keras":
            input_config = lambda_model
        elif input_format == "json":
            input_config = lambda_model.to_json()
        elif input_format == "dict":
            input_config = lambda_model.get_config()
        elif input_format == "path":
            path_ = pathlib.Path(str(tmpdir))
            input_config = path_ / "model.json"
            with input_config.open("w") as f_a:
                json.dump(lambda_model.to_json(), f_a)
        else:
            raise ValueError()

        output = replace_lambda_in_config(input_config, output_format=output_format)

        # check types
        if output_format == "keras":
            assert isinstance(output, keras.Model)
            new_model = output
        elif output_format == "json":
            assert isinstance(output, str)
            new_model = keras.models.model_from_json(
                output,
            )
        elif output_format == "dict":
            assert isinstance(output, dict)
            new_model = keras.Model.from_config(
                output,
                custom_objects={
                    "ExtractMoving": ExtractMoving,
                    "NoOp": NoOp,
                },
            )

        shape_input = keras.backend.int_shape(lambda_model.input)[1:]
        x = np.random.random((2, *shape_input))

        assert np.allclose(lambda_model.predict(x), new_model.predict(x))

    def test_incorrect_input(self, lambda_model):
        # incorrect input type
        with pytest.raises(TypeError):
            replace_lambda_in_config(1)

        # incorret input path
        with pytest.raises(ValueError):
            replace_lambda_in_config(pathlib.Path.cwd() / "fake.wrong")

        # incorrect output type
        with pytest.raises(TypeError):
            replace_lambda_in_config(lambda_model, output_format="fake")

    def test_incorrect_layer_config(self, lambda_model):

        correct_config = lambda_model.get_config()
        missing_class_name = deepcopy(correct_config)
        missing_config = deepcopy(correct_config)
        invalid_name = deepcopy(correct_config)

        # prep
        del missing_class_name["layers"][0]["class_name"]
        del missing_config["layers"][1]["config"]
        invalid_name["layers"][2]["config"]["name"] = "aaaaa"

        with pytest.raises(KeyError):
            replace_lambda_in_config(missing_class_name)

        with pytest.raises(KeyError):
            replace_lambda_in_config(missing_config)

        with pytest.raises(KeyError):
            replace_lambda_in_config(invalid_name)
