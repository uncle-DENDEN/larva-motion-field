import six
from collections import Iterable
import numbers
import six
import yaml
from pathlib import Path


def _mkdir_if_valid(name, value):
    if '_path' in name:
        Path(value).mkdir(parents=True, exist_ok=True)


def _cast_to_type_if_compatible(name, param_type, value):
    """
        Cast hparam to the provided type, if compatible.
        Args:
        name: Name of the hparam to be cast.
        param_type: The type of the hparam.
        value: The value to be cast, if compatible.
        Returns:
        The result of casting `value` to `param_type`.
        Raises:
        ValueError: If the type of `value` is not compatible with param_type.
          * If `param_type` is a string type, but `value` is not.
          * If `param_type` is a boolean, but `value` is not, or vice versa.
          * If `param_type` is an integer type, but `value` is not.
          * If `param_type` is a float type, but `value` is not a numeric type.
    """
    fail_msg = (
            "Could not cast hparam '%s' of type '%s' from value %r" %
            (name, param_type, value))

    # Some callers use None, for which we can't do any casting/checking. :(
    if issubclass(param_type, type(None)):
        return value

    # Avoid converting a non-string type to a string.
    if (issubclass(param_type, (six.string_types, six.binary_type)) and
            not isinstance(value, (six.string_types, six.binary_type))):
        raise ValueError(fail_msg)

    # Avoid converting a number or string type to a boolean or vice versa.
    if issubclass(param_type, bool) != isinstance(value, bool):
        raise ValueError(fail_msg)

    # Avoid converting float to an integer (the reverse is fine).
    if (issubclass(param_type, numbers.Integral) and
            not isinstance(value, numbers.Integral)):
        raise ValueError(fail_msg)

    # Avoid converting a non-numeric type to a numeric type.
    if (issubclass(param_type, numbers.Number) and
            not isinstance(value, numbers.Number)):
        raise ValueError(fail_msg)

    return param_type(value)


class HParams(object):
    """
        Class to hold a set of hyperparameters as name-value pairs.
        A `HParams` object holds hyperparameters used to build and train a model,
        such as the number of hidden units in a neural net layer or the learning rate
        to use when training.
        You first create a `HParams` object by specifying the names and values of the
        hyperparameters.
        To make them easily accessible the parameter names are added as direct
        attributes of the class.  A typical usage is as follows:
        ```python
        # Create a HParams object specifying names and values of the model
        # hyperparameters:
        hparams = HParams(learning_rate=0.1, num_hidden_units=100)
        # The hyperparameter are available as attributes of the HParams object:
        hparams.learning_rate ==> 0.1
        hparams.num_hidden_units ==> 100
        ```
        Hyperparameters have type, which is inferred from the type of their value
        passed at construction type.   The currently supported types are: integer,
        float, boolean, string, and list of integer, float, boolean, or string.
        ```
    """

    _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

    def __init__(self, model_structure=None, **kwargs):
        # Register the hyperparameters and their type in _hparam_types.
        # This simplifies the implementation of parse().
        # _hparam_types maps the parameter name to a tuple (type, bool).
        # The type value is the type of the parameter for scalar hyperparameters,
        # or the type of the list elements for multidimensional hyperparameters.
        # The bool value is True if the value is a list, False otherwise.
        self._hparam_types = {}
        self._model_structure = model_structure
        for name, value in six.iteritems(kwargs):
            self.add_hparam(name, value)

    def add_hparam(self, name, value):
        # Keys in kwargs are unique, but 'name' could the name of a pre-existing
        # attribute of this object.  In that case we refuse to use it as a
        # hyperparameter name.
        if getattr(self, name, None) is not None:
            raise ValueError('Hyperparameter name is reserved: %s' % name)
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(
                    'Iterable hyperparameters cannot be empty: %s' % name)
            self._hparam_types[name] = (type(value[0]), True)
        elif isinstance(value, str):
            _mkdir_if_valid(name, value)
            self._hparam_types[name] = (str, False)
        else:
            self._hparam_types[name] = (type(value), False)
        setattr(self, name, value)

    def set_hparam(self, name, value):
        """
        Set the value of an existing hyperparameter.
        """
        param_type, is_list = self._hparam_types[name]
        if isinstance(value, list):
            if not is_list:
                raise ValueError(
                    'Must not pass a list for single-valued parameter: %s' % name)
            setattr(self, name, [
                _cast_to_type_if_compatible(name, param_type, v) for v in value])
        elif isinstance(value, tuple):
            if param_type is not dict:
                raise ValueError(
                    'Must not pass a key-value pair for non-dict parameter: %s.' % name)
            curr_dict = getattr(self, name)
            update = {value[0]: value[1]}
            curr_dict.update(update)
            setattr(self, name, curr_dict)
        else:
            if is_list:
                raise ValueError(
                    'Must pass a list for Multi-valued parameter: %s' % name)
            if param_type is dict:
                raise ValueError(
                    'Must pass a key-value pair for dict parameter: %s' % name)
            setattr(self, name, _cast_to_type_if_compatible(name, param_type, value))

    def del_hparam(self, name):
        if hasattr(self, name):
            delattr(self, name)
            del self._hparam_types[name]

    @classmethod
    def from_yaml(cls, yaml_path):
        with open("{}".format(yaml_path), "r") as stream:
            try:
                hparams = yaml.load(stream, yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)

        # hp_vals = hparams.values()
        return HParams(**hparams)
