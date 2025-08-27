import inspect
import types
from importlib import import_module
from typing import List, Type

from ormatic.dao import to_dao

from semantic_world.orm.ormatic_interface import RosMessageMixin, PrefixedNameDAO, Base
from semantic_world.prefixed_name import PrefixedName


def main():
    # 2) Only subclasses defined in a specific module/file
    def ros_mixins_in_module(module: str | types.ModuleType) -> List[Type]:
        mod = import_module(module) if isinstance(module, str) else module
        out: List[Type] = []
        for name, obj in vars(mod).items():
            if isinstance(obj, type) and obj is not RosMessageMixin and issubclass(obj, RosMessageMixin):
                if obj.__module__ == mod.__name__:  # defined in that file
                    out.append(obj)
        try:
            out.sort(key=lambda c: inspect.getsourcelines(c)[1])
        except OSError:
            out.sort(key=lambda c: c.__name__)
        return out

    mixins = ros_mixins_in_module("semantic_world.orm.ormatic_interface")

    PrefixedNameDAO.to_ros_message_file('/home/luca-krohm/work/semantic_world/tmp')

    [clazz.to_ros_message_file('/home/luca-krohm/work/semantic_world/tmp') for clazz in mixins if clazz != Base]

    name = to_dao(PrefixedName(name="name", prefix="test_prefix"))
    print(name.get_ros_message())

if __name__ == "__main__":
    main()
