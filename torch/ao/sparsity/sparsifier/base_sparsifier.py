import abc
import copy
import warnings
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn.utils import parametrize

from .utils import FakeSparsity, module_to_fqn, fqn_to_module, get_arg_info_from_tensor_fqn

__all__ = ["BaseSparsifier"]

SUPPORTED_MODULES = {
    nn.Linear
}

KEYS_NOT_IN_STATE_DICT = ['module', 'module_fqn', 'tensor_name']

# TODO update desc with new config args
class BaseSparsifier(abc.ABC):
    r"""Base class for all sparsifiers.

    Abstract methods that need to be implemented:

    - update_mask: Function to compute a new mask for all keys in the
        `groups`.

    Args:
        - model [nn.Module]: model to configure. The model itself is not saved
            but used for the state_dict saving / loading.
        - config [list]: configuration elements should be a dict map that includes
            `tensor_fqn` of tensors to sparsify
        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.

    Example::

        >>> config = [{'tensor_fqn': 'layer1.weight', {'tensor_fqn': 'linear2.weight2', 'sparsity_level': 0.5}]
        >>> defaults = {'sparsity_level': 0.7}
        >>> # model.layer1.weight will have `sparsity_level` = 0.7 (getting default)
        >>> sparsifier = BaseSparsifier(config, defaults)
    """
    def __init__(self, defaults):
        super().__init__()
        self.defaults = defaults
        if self.defaults is None:
            self.defaults = dict()

        self.state: Dict[str, Dict] = defaultdict(dict)
        self.groups = []
        self.enable_mask_update = True

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'groups': self.groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, sparse_args in enumerate(self.groups):
            module = sparse_args['module']
            format_string += '\n'
            format_string += f'\tModule Group {i}\n'
            format_string += f'\t    module: {module}\n'
            for key in sorted(sparse_args.keys()):
                if key == 'module':
                    continue
                format_string += f'\t    {key}: {sparse_args[key]}\n'
        format_string += ')'
        return format_string

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains:
        * state - current state of the sparsification.
        * groups - a list containing all sparsity configuration groups
            with the key 'tensor_fqn' specifying the path to the sparsified tensor within a model

        TODO: Need a clean way of loading the state of the "prepared" module
        """


        groups = [
            dict(filter(lambda key_value: key_value[0] not in KEYS_NOT_IN_STATE_DICT , mg.items()))
            for mg in self.groups
        ]

        return {
            'state': self.state,
            'groups': groups,
        }

    def load_state_dict(self, state_dict, strict=True):
        groups = copy.deepcopy(state_dict['groups'])
        states = state_dict['state']
        for tensor_fqn, s in states.items():
            arg_info = get_arg_info_from_tensor_fqn(self.model, tensor_fqn)
            module = arg_info['module']
            tensor_name = arg_info['tensor_name']
            if strict and module is None:
                raise RuntimeError(f'Error loading {tensor_fqn} into the model')

            found = False
            for p in module.parametrizations[tensor_name]:
                if isinstance(p, FakeSparsity):
                    found = True
                    break
            if not found:
                p = FakeSparsity(torch.ones(getattr(module, tensor_name).shape))
                parametrize.register_parametrization(module, tensor_name, p)
            if s.get('mask', None) is not None:
                mask = s.pop('mask')
                p.mask = mask

            for mg in groups:
                if mg['tensor_fqn'] == tensor_fqn:
                    mg.update(arg_info)
        self.__setstate__({'state': states, 'groups': groups})

    def make_config_from_model(self, model, SUPPORTED_MODULES=SUPPORTED_MODULES, NEEDS_ZEROS=None):
        self.config = []
        stack = [model]
        while stack:
            module = stack.pop()
            for name, child in module.named_children():
                if type(child) in SUPPORTED_MODULES:
                    self.config.append({'tensor_fqn': module_to_fqn(model, child) + '.weight'})
                else:
                    stack.append(child)

    def prepare(self, model, config):
        r"""Prepares a model, by adding the parametrizations.

        Note::

            The model is modified inplace. If you need to preserve the original
            model, use copy.deepcopy.
        """
        self.model = model  # TODO: Need to figure out how to load without this.
        self.config = config

        # If no config -- try getting all the supported layers
        if self.config is None:
            self.make_config_from_model(model)

        # TODO: Remove the configuration by reference ('module')
        for module_config in self.config:
            if isinstance(module_config, nn.Module):
                warnings.warn("config elements should be dicts not modules")
                module_config = {'module': module_config}
            local_args = copy.deepcopy(self.defaults)
            local_args.update(module_config)
            # Make sure there is at least one way of handling the model
            tensor_fqn = local_args.get('tensor_fqn', None)

            if tensor_fqn is None:
                warnings.warn(
                    "tensor_fqn is a required argument in the sparsity config"
                    "and support for `module` and `module_fqn` will be deprecated"
                )
                module = local_args.get('module', None)
                module_fqn = local_args.get('module_fqn', None)

                if module is None and module_fqn is None:
                    # No module given for this group
                    raise ValueError('Either `tensor_fqn` or `module` or `module_fqn` must be specified!')
                elif module is None:
                    # FQN is given
                    module = fqn_to_module(model, module_fqn)
                elif module_fqn is None:
                    # Module is given
                    module_fqn = module_to_fqn(model, module)
                else:
                    # Both Module and FQN are given
                    module_from_fqn = fqn_to_module(model, module_fqn)
                    assert module is module_from_fqn, \
                        'Given both `module` and `fqn`, it is expected them to ' \
                        'refer to the same thing!'
                if module_fqn and module_fqn[0] == '.':
                    module_fqn = module_fqn[1:]
                local_args['module_fqn'] = module_fqn
                local_args['module'] = module
                local_args['tensor_fqn'] = module_fqn + '.weight'
                local_args['tensor_name'] = 'weight'
            else:
                info_from_tensor_fqn = get_arg_info_from_tensor_fqn(model, tensor_fqn)

                # check that whatever was put into local_args agrees with what was obtained
                # from tensor_fqn
                for key in info_from_tensor_fqn.keys():
                    if key in local_args:
                        # info_from_tensor_fqn will chop leading '.' from tensor_fqn so ignore that
                        assert key == 'tensor_fqn' or info_from_tensor_fqn[key] == local_args[key], (
                            "Given both `{}` and `tensor_fqn`, it is expected them to "
                            "agree!".format(key)
                        )
                local_args.update(info_from_tensor_fqn)

            self.groups.append(local_args)
        self._prepare()

    def _prepare(self, *args, **kwargs):
        r"""Adds mask parametrization to the layer weight
        """
        for config in self.groups:
            module = config['module']
            tensor_name = config['tensor_name']
            parametrization = config.get('parametrization', FakeSparsity)
            mask = config.get('mask', torch.ones_like(getattr(module, tensor_name)))
            self.state[config['tensor_fqn']]['mask'] = mask
            parametrize.register_parametrization(module, tensor_name, parametrization(mask))

    def squash_mask(self,
                    params_to_keep: Optional[Tuple[str, ...]] = None,
                    params_to_keep_per_layer: Optional[Dict[str, Tuple[str, ...]]] = None,
                    *args, **kwargs):
        r"""Squashes the sparse masks into the appropriate tensors.

        If either the `params_to_keep` or `params_to_keep_per_layer` is set,
        the module will have a `sparse_params` dict attached to it.

        Args:
            params_to_keep: List of keys to save in the module or a dict
                            representing the modules and keys that will have
                            sparsity parameters saved
            params_to_keep_per_layer: Dict to specify the params that should be
                            saved for specific layers. The keys in the dict
                            should be the module fqn, while the values should
                            be a list of strings with the names of the variables
                            to save in the `sparse_params`

        Examples:
            >>> # Don't save any sparse params
            >>> sparsifier.squash_mask()
            >>> hasattr(model.submodule1, 'sparse_params')
            False

            >>> # Keep sparse params per layer
            >>> sparsifier.squash_mask(
            ...     params_to_keep_per_layer={
            ...         'submodule1.linear1': ('foo', 'bar'),
            ...         'submodule2.linear42': ('baz',)
            ...     })
            >>> print(model.submodule1.linear1.sparse_params)
            {'foo': 42, 'bar': 24}
            >>> print(model.submodule2.linear42.sparse_params)
            {'baz': 0.1}

            >>> # Keep sparse params for all layers
            >>> sparsifier.squash_mask(params_to_keep=('foo', 'bar'))
            >>> print(model.submodule1.linear1.sparse_params)
            {'foo': 42, 'bar': 24}
            >>> print(model.submodule2.linear42.sparse_params)
            {'foo': 42, 'bar': 24}

            >>> # Keep some sparse params for all layers, and specific ones for
            >>> # some other layers
            >>> sparsifier.squash_mask(
            ...     params_to_keep=('foo', 'bar'),
            ...     params_to_keep_per_layer={
            ...         'submodule2.linear42': ('baz',)
            ...     })
            >>> print(model.submodule1.linear1.sparse_params)
            {'foo': 42, 'bar': 24}
            >>> print(model.submodule2.linear42.sparse_params)
            {'foo': 42, 'bar': 24, 'baz': 0.1}
        """
        for config in self.groups:
            module = config['module']
            tensor_name = config['tensor_name']
            parametrize.remove_parametrizations(module, tensor_name,
                                                leave_parametrized=True)
            sparse_params = dict()
            if params_to_keep is not None:
                global_params = {k: config[k] for k in params_to_keep}
                sparse_params.update(global_params)
            if params_to_keep_per_layer is not None:
                params = params_to_keep_per_layer.get(config['module_fqn'], None)
                if params is not None:
                    per_layer_params = {k: config[k] for k in params}
                    sparse_params.update(per_layer_params)
            if sparse_params:
                module.sparse_params = sparse_params

    def convert(self):
        # TODO: Call the torch.ao.utils.convert in here
        raise NotImplementedError('`convert` is not implemented. Please, use '
                                  '`torch.ao.utils.convert` instead.')

    def step(self, use_path=True):
        if not self.enable_mask_update:
            return
        with torch.no_grad():
            for config in self.groups:
                self.update_mask(**config)

    @abc.abstractmethod
    def update_mask(self, module, tensor_name, **kwargs):
        pass
