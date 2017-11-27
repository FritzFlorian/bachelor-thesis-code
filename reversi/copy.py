"""Own alternative to the built in copy as it is really slow."""
import pickle
import copy as old_copy


def copy(obj):
    return old_copy.copy(obj)


def deepcopy(obj):
    return pickle.loads(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
