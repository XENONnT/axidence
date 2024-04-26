from immutabledict import immutabledict
from tqdm import tqdm
import strax
import straxen

from axidence import SaltingEvents, SaltingPeaks
from axidence import (
    SaltingPeakProximity,
    SaltingPeakShadow,
    SaltingPeakAmbience,
    SaltingPeakSEDensity,
)
from axidence import (
    SaltedEvents,
    SaltedEventBasics,
    SaltedEventShadow,
    SaltedEventAmbience,
    SaltedEventSEDensity,
)
from axidence import EventBuilding


def unsalted_context(**kwargs):
    return straxen.contexts.xenonnt_online(_database_init=False, **kwargs)


@strax.Context.add_method
def salt_to_context(self):
    self.register(
        (
            SaltingEvents,
            SaltingPeaks,
            SaltingPeakProximity,
            SaltingPeakShadow,
            SaltingPeakAmbience,
            SaltingPeakSEDensity,
            SaltedEvents,
            SaltedEventBasics,
            SaltedEventShadow,
            SaltedEventAmbience,
            SaltedEventSEDensity,
            EventBuilding,
        )
    )


@strax.Context.add_method
def plugin_factory(st, data_type, suffixes):
    plugin = st._plugin_class_registry[data_type]

    new_plugins = []
    p = st._get_plugins((data_type,), run_id="0")[data_type]

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
    snakes = ["_" + strax.camel_to_snake(suffix) for suffix in suffixes]
    for k in st._plugin_class_registry.keys():
        for s in snakes:
            if s in k:
                raise ValueError(f"{k} with suffix {s} is already registered!")
    plugins_collection = []
    for k in tqdm(st._plugin_class_registry.keys(), disable=tqdm_disable):
        plugins_collection += st.plugin_factory(k, suffixes)

    st.register(plugins_collection)
