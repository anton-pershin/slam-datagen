from typing import TypeAlias, Union

NestedStrDict: TypeAlias = dict[str, Union["NestedStrDict", str]]
