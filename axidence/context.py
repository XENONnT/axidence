from immutabledict import immutabledict
from tqdm import tqdm
import strax
import straxen

from axidence import EventsSalting, PeaksSalted
from axidence import (
    PeakProximitySalted,
    PeakShadowSalted,
    PeakAmbienceSalted,
    PeakSEDensitySalted,
)
from axidence import (
    EventsSalted,
    EventBasicsSalted,
    EventShadowSalted,
    EventAmbienceSalted,
    EventSEDensitySalted,
    EventBuilding,
)
from axidence import (
    IsolatedS1Mask,
    IsolatedS2Mask,
    IsolatedS1,
    IsolatedS2,
    PeaksPaired,
    PeakProximityPaired,
)


def ordinary_context(**kwargs):
    """Return a straxen context without paring and salting."""
    return straxen.contexts.xenonnt_online(_database_init=False, **kwargs)


@strax.Context.add_method
def plugin_factory(st, data_type, suffixes):
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

            def do_compute(self, chunk_i=None, **kwargs):
                # remove the suffix from the keys
                new_keys = [k.replace(self.suffix, "") for k in kwargs.keys()]
                new_kwargs = dict(zip(new_keys, kwargs.values()))
                return super().do_compute(chunk_i=chunk_i, **new_kwargs)

        # need to be compatible with strax.camel_to_snake
        # https://github.com/AxFoundation/strax/blob/7da9a2a6375e7614181830484b322389986cf064/strax/context.py#L324
        new_plugin.__name__ = plugin.__name__ + suffix

        # assign the same attributes as the original plugin
        if hasattr(p, "depends_on"):
            new_plugin.depends_on = tuple(d + snake for d in p.depends_on)
        else:
            raise RuntimeError

        if hasattr(p, "provides"):
            new_plugin.provides = tuple(p + snake for p in p.provides)
        else:
            snake_name = strax.camel_to_snake(new_plugin.__name__)
            new_plugin.provides = (snake_name,)

        if hasattr(p, "data_kind"):
            if isinstance(p.data_kind, (dict, immutabledict)):
                keys = [k + snake for k in p.data_kind.keys()]
                values = [v + snake for v in p.data_kind.values()]
                new_plugin.data_kind = immutabledict(zip(keys, values))
            else:
                new_plugin.data_kind = p.data_kind + snake
        else:
            raise RuntimeError(f"data_kind is not defined for instance of {plugin.__name__}")

        if hasattr(p, "save_when"):
            if isinstance(p.save_when, (dict, immutabledict)):
                keys = [k + snake for k in p.save_when]
                new_plugin.save_when = immutabledict(zip(keys, p.save_when.values()))
            else:
                new_plugin.save_when = p.save_when + snake
        else:
            raise RuntimeError(f"save_when is not defined for instance of {plugin.__name__}")

        if isinstance(p.dtype, (dict, immutabledict)):
            new_plugin.dtype = dict(zip([k + snake for k in p.dtype.keys()], p.dtype.values()))
        else:
            new_plugin.dtype = p.dtype

        new_plugins.append(new_plugin)
    return new_plugins


@strax.Context.add_method
def replication_tree(st, suffixes=["Paired", "Salted"], tqdm_disable=True):
    """Replicate the dependency tree.

    The plugins in the new tree will have the suffixed depends_on,
    provides and data_kind as the plugins in original tree.
    """
    snakes = ["_" + strax.camel_to_snake(suffix) for suffix in suffixes]
    for k in st._plugin_class_registry.keys():
        for s in snakes:
            if s in k:
                raise ValueError(f"{k} with suffix {s} is already registered!")
    plugins_collection = []
    for k in tqdm(st._plugin_class_registry.keys(), disable=tqdm_disable):
        plugins_collection += st.plugin_factory(k, suffixes)

    st.register(plugins_collection)


@strax.Context.add_method
def _salt_to_context(self):
    """Register the salted plugins to the context."""
    self.register(
        (
            EventsSalting,
            PeaksSalted,
            PeakProximitySalted,
            PeakShadowSalted,
            PeakAmbienceSalted,
            PeakSEDensitySalted,
            EventsSalted,
            EventBasicsSalted,
            EventShadowSalted,
            EventAmbienceSalted,
            EventSEDensitySalted,
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
        )
    )


@strax.Context.add_method
def salt_to_context(st, tqdm_disable=True):
    """Register the salted plugins to the context."""
    st.register((EventBuilding,))
    st.replication_tree(suffixes=["Salted"], tqdm_disable=tqdm_disable)
    st._salt_to_context()


@strax.Context.add_method
def salt_and_pair_to_context(st, tqdm_disable=True):
    """Register the salted and paired plugins to the context."""
    st.register((EventBuilding,))
    st.replication_tree(tqdm_disable=tqdm_disable)
    st._salt_to_context()
    st._pair_to_context()
