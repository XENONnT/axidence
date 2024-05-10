from immutabledict import immutabledict
from tqdm import tqdm
import numpy as np
import strax
from strax import LoopPlugin, CutPlugin, CutList
import straxen
from straxen import EventBasics, EventInfoDouble

from axidence import RunMeta, EventsSalting, PeaksSalted
from axidence import (
    PeakProximitySalted,
    PeakShadowSalted,
    PeakAmbienceSalted,
    PeakNearestTriggeringSalted,
    PeakSEDensitySalted,
)
from axidence import (
    EventsSalted,
    EventBasicsSalted,
    EventShadowSalted,
    EventAmbienceSalted,
    EventNearestTriggeringSalted,
    EventSEDensitySalted,
    EventsCombine,
    MainS1Trigger,
    MainS2Trigger,
    EventBuilding,
)
from axidence import (
    IsolatedS1Mask,
    IsolatedS2Mask,
    IsolatedS1,
    IsolatedS2,
    PeaksPaired,
    PeakProximityPaired,
    PeakPositionsPaired,
    EventInfosPaired,
    PairingExists,
)

export, __all__ = strax.exporter()
__all__.extend(["default_assign_attributes", "default_assign_appended_attributes"])


default_assign_attributes = {
    EventBasics: ["peak_properties", "posrec_save"],
    EventInfoDouble: ["input_dtype"],
}


allow_hyperrun_suffix = ["Paired"]


default_assign_appended_attributes: dict = {}


@export
def ordinary_context(**kwargs):
    """Return a straxen context without paring and salting."""
    return straxen.contexts.xenonnt_online(_database_init=False, **kwargs)


@export
def assign_plugin_attributes(
    new_plugin,
    old_plugin,
    old_instance,
    suffix,
    snake,
    assign_attributes=None,
    assign_appended_attributes=None,
):
    # need to be compatible with strax.camel_to_snake
    # https://github.com/AxFoundation/strax/blob/7da9a2a6375e7614181830484b322389986cf064/strax/context.py#L324
    new_plugin.__name__ = old_plugin.__name__ + suffix

    # assign the attributes from the original plugin
    if assign_attributes:
        for k, v in assign_attributes.items():
            if not issubclass(old_plugin, k):
                continue
            for attr in v:
                setattr(new_plugin, attr, getattr(old_instance, attr))

    if assign_appended_attributes:
        for k, v in assign_appended_attributes.items():
            if not issubclass(old_plugin, k):
                continue
            for attr in v:
                setattr(new_plugin, attr, getattr(old_instance, attr) + snake)

    # assign the same attributes as the original plugin
    if hasattr(old_instance, "depends_on"):
        new_plugin.depends_on = tuple(d + snake for d in old_instance.depends_on)
    else:
        raise RuntimeError(f"depends_on is not defined for instance of {old_plugin.__name__}")

    if hasattr(old_instance, "provides"):
        new_plugin.provides = tuple(p + snake for p in old_instance.provides)
    else:
        snake_name = strax.camel_to_snake(new_plugin.__name__)
        new_plugin.provides = (snake_name,)

    if hasattr(old_instance, "data_kind"):
        if isinstance(old_instance.data_kind, (dict, immutabledict)):
            keys = [k + snake for k in old_instance.data_kind.keys()]
            values = [v + snake for v in old_instance.data_kind.values()]
            new_plugin.data_kind = immutabledict(zip(keys, values))
        else:
            new_plugin.data_kind = old_instance.data_kind + snake
    else:
        raise RuntimeError(f"data_kind is not defined for instance of {old_plugin.__name__}")

    if hasattr(old_instance, "save_when"):
        if isinstance(old_instance.save_when, (dict, immutabledict)):
            keys = [k + snake for k in old_instance.save_when]
            new_plugin.save_when = immutabledict(zip(keys, old_instance.save_when.values()))
        else:
            new_plugin.save_when = old_instance.save_when + snake
    else:
        raise RuntimeError(f"save_when is not defined for instance of {old_plugin.__name__}")

    if isinstance(old_instance.dtype, (dict, immutabledict)):
        new_plugin.dtype = dict(
            zip([k + snake for k in old_instance.dtype.keys()], old_instance.dtype.values())
        )
        # some plugins like EventAreaPerChannel also uses self.dtype in compute
        new_plugin.dtype.update(old_instance.dtype)
    else:
        new_plugin.dtype = old_instance.dtype

    if isinstance(old_instance, CutPlugin):
        if hasattr(old_instance, "cut_name"):
            new_plugin.cut_name = old_instance.cut_name + snake
        else:
            raise RuntimeError(f"cut_name is not defined for instance of {old_plugin.__name__}")

    if isinstance(old_instance, CutList):
        if hasattr(old_instance, "accumulated_cuts_string"):
            new_plugin.accumulated_cuts_string = old_instance.accumulated_cuts_string + snake
        else:
            raise RuntimeError(
                f"accumulated_cuts_string is not defined for instance of {old_plugin.__name__}"
            )

    if isinstance(old_instance, CutPlugin) or isinstance(old_instance, CutList):
        # this will make CutList.cuts to be invalid
        new_plugin.dtype = np.dtype(
            [
                ((d[0][0], d[0][1] + snake), d[1]) if d[0][1] not in ["time", "endtime"] else d
                for d in new_plugin.dtype.descr
            ]
        )

    if hasattr(old_instance, "loop_over"):
        new_plugin.loop_over = old_instance.loop_over + snake

    if suffix in allow_hyperrun_suffix:
        new_plugin.allow_hyperrun = True

    return new_plugin


@export
def keys_detach_suffix(kwargs, snake):
    # remove the suffix from the keys
    new_keys = [k.replace(snake, "") for k in kwargs.keys()]
    new_kwargs = dict(zip(new_keys, kwargs.values()))
    return new_kwargs


@export
def keys_attach_suffix(kwargs, snake):
    # remove the suffix from the keys
    new_keys = [k + snake for k in kwargs.keys()]
    new_kwargs = dict(zip(new_keys, kwargs.values()))
    return new_kwargs


@strax.Context.add_method
def plugin_factory(
    st, data_type, suffixes, assign_attributes=None, assign_appended_attributes=None
):
    """Create new plugins inheriting from the plugin which provides
    data_type."""
    plugin = st._plugin_class_registry[data_type]

    new_plugins = []
    p = st._Context__get_plugin(run_id="0", data_type=data_type)

    for suffix in suffixes:
        snake = "_" + strax.camel_to_snake(suffix)

        class new_plugin(plugin):
            suffix = snake

            def infer_dtype(self):
                # some plugins like PulseProcessing uses self.deps in infer_dtype,
                # which will cause error because the dependency tree changes
                # https://github.com/XENONnT/straxen/blob/b4910e560a6a7f11288a4368816e692c26f8bc73/straxen/plugins/records/records.py#L142
                # so we assign the dtype manually and raise error in infer_dtype method
                raise RuntimeError

            if not issubclass(plugin, LoopPlugin):

                def _fix_output(self, result, start, end, _dtype=None):
                    if self.multi_output and _dtype is None:
                        result = keys_attach_suffix(result, self.suffix)
                        return {
                            d: super(plugin, self)._fix_output(result[d], start, end, _dtype=d)
                            for d in self.provides
                        }
                    else:
                        return super()._fix_output(result, start, end, _dtype=_dtype)

                def do_compute(self, chunk_i=None, **kwargs):
                    return super().do_compute(
                        chunk_i=chunk_i, **keys_detach_suffix(kwargs, self.suffix)
                    )

            else:

                def compute_loop(self, base_chunk, **kwargs):
                    result = super().compute_loop(
                        base_chunk, **keys_detach_suffix(kwargs, self.suffix)
                    )
                    if self.multi_output:
                        return keys_attach_suffix(result, self.suffix)
                    else:
                        return result

            if issubclass(plugin, CutPlugin):

                def cut_by(self, **kwargs):
                    return super().cut_by(**keys_detach_suffix(kwargs, self.suffix))

        new_plugin = assign_plugin_attributes(
            new_plugin,
            plugin,
            p,
            suffix,
            snake,
            assign_attributes=assign_attributes,
            assign_appended_attributes=assign_appended_attributes,
        )

        new_plugins.append(new_plugin)
    return new_plugins


@strax.Context.add_method
def replication_tree(
    st,
    suffixes=["Paired", "Salted"],
    assign_attributes=None,
    assign_appended_attributes=None,
    tqdm_disable=True,
):
    """Replicate the dependency tree.

    The plugins in the new tree will have the suffixed depends_on,
    provides and data_kind as the plugins in original tree.
    """
    # this is due to some features are assigned in `infer_dtype` of the original plugins:
    # https://github.com/XENONnT/straxen/blob/e555c7dcada2743d2ea627ea49df783e9dba40e3/straxen/plugins/events/event_basics.py#L69
    if assign_attributes is None:
        assign_attributes = default_assign_attributes

    if assign_appended_attributes is None:
        assign_appended_attributes = default_assign_appended_attributes

    snakes = ["_" + strax.camel_to_snake(suffix) for suffix in suffixes]
    for k in st._plugin_class_registry.keys():
        for s in snakes:
            if k.endswith(s):
                raise ValueError(f"{k} with suffix {s} is already registered!")
    plugins_collection = []
    for k in tqdm(st._plugin_class_registry.keys(), disable=tqdm_disable):
        plugins_collection += st.plugin_factory(
            k,
            suffixes,
            assign_attributes=assign_attributes,
            assign_appended_attributes=assign_appended_attributes,
        )

    st.register(plugins_collection)


@strax.Context.add_method
def _salt_to_context(self):
    """Register the salted plugins to the context."""
    self.register(
        (
            RunMeta,
            EventsSalting,
            PeaksSalted,
            PeakProximitySalted,
            PeakShadowSalted,
            PeakAmbienceSalted,
            PeakNearestTriggeringSalted,
            PeakSEDensitySalted,
            EventsSalted,
            EventBasicsSalted,
            EventShadowSalted,
            EventAmbienceSalted,
            EventNearestTriggeringSalted,
            EventSEDensitySalted,
            EventsCombine,
        )
    )


@strax.Context.add_method
def _pair_to_context(self):
    """Register the paired plugins to the context."""
    self.register(
        (
            IsolatedS1Mask,
            IsolatedS2Mask,
            IsolatedS1,
            IsolatedS2,
            PeaksPaired,
            PeakProximityPaired,
            PeakPositionsPaired,
            EventInfosPaired,
            PairingExists,
        )
    )


@strax.Context.add_method
def salt_to_context(st, assign_attributes=None, tqdm_disable=True):
    """Register the salted plugins to the context."""
    st.register((MainS1Trigger, MainS2Trigger, EventBuilding))
    st.replication_tree(
        suffixes=["Salted"], assign_attributes=assign_attributes, tqdm_disable=tqdm_disable
    )
    st._salt_to_context()


@strax.Context.add_method
def salt_and_pair_to_context(st, assign_attributes=None, tqdm_disable=True):
    """Register the salted and paired plugins to the context."""
    st.register((MainS1Trigger, MainS2Trigger, EventBuilding))
    st.replication_tree(assign_attributes=assign_attributes, tqdm_disable=tqdm_disable)
    st._salt_to_context()
    st._pair_to_context()
