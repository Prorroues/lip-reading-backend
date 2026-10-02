import importlib


# def dynamic_import(import_path, alias=dict()):
#     """dynamic import module and class
#
#     :param str import_path: syntax 'module_name:class_name'
#         e.g., 'espnet.transform.add_deltas:AddDeltas'
#     :param dict alias: shortcut for registered class
#     :return: imported class
#     """
#     print(import_path, "hjdhdjh")
#     if import_path not in alias and ":" not in import_path:
#         raise ValueError(
#             "import_path should be one of {} or "
#             'include ":", e.g. "espnet.transform.add_deltas:AddDeltas" : '
#             "{}".format(set(alias), import_path)
#         )
#     if ":" not in import_path:
#         import_path = alias[import_path]
#     print(import_path, "hjdhdjh")
#     module_name, objname = import_path.split(":")
#     m = importlib.import_module(module_name)
#     return getattr(m, objname)


import importlib

def dynamic_import(module_name, alias_dict=None):
    if alias_dict and module_name in alias_dict:
        module_name = alias_dict[module_name]

    if ":" in module_name:
        module_path, class_name = module_name.split(":")
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    else:
        if module_name.startswith("."):
            module = importlib.import_module(module_name, package=__name__)
        else:
            module = importlib.import_module(module_name)
        return module


