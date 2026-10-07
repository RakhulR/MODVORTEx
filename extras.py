# -*- coding: utf-8 -*-
"""
Local helpers vendored from the external ``mymodule`` package so this project
has no dependency on it.

Vendored from ``mymodule==0.0.4``:
- ``decorate_all_methods`` from ``mymodule/utils.py``
- ``exception_handler`` and its internal helper ``show_error_dialog``
  from ``mymodule/exceptions.py``

The implementations below are verbatim copies; only this header comment was
added. If the upstream package changes, sync the change here manually.
"""

from types import FunctionType
from typing import Callable, Optional, Type
from functools import wraps

from PyQt5.QtWidgets import QMessageBox, QWidget


def decorate_all_methods(decorator: Callable) -> Callable:
    def class_decorator(cls: Type) -> Type:
        for attr_name, attr_value in cls.__dict__.items():
            if isinstance(attr_value, FunctionType) and not attr_name.startswith("__"):
                setattr(cls, attr_name, decorator(attr_value))
        return cls
    return class_decorator


def exception_handler(func: Callable) -> Callable:
    """
    A decorator that wraps the passed in function and executes it.
    If an exception occurs, it catches the exception and calls the
    show_error_dialog function with the exception.

    Args:
        func (function): The function to wrap.

    Returns:
        function: The wrapper function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # func_sig = signature(func)
        # func_params = func_sig.parameters
        func_params_len = func.__code__.co_argcount
        try:

            if not args:
                return func(*args, **kwargs)

            # If the first argument is a QWidget, adjust the args based on the function's signature
            elif isinstance(args[0], QWidget): # checks if this function is widget since signals form those events like buttonpress optionally pass something the function
                # Filter out the args based on the number of parameters the function expects
                if func_params_len < 2: # any function with more than one arguments will be managed by lambda expression

                    return func(*args[:1], **kwargs)

                    # filtered_args = args[:len(func_params)]
                    # return func(*filtered_args, **kwargs)
                else:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        except Exception as ase:
            if isinstance(args[0], QWidget):
                show_error_dialog(error_message=ase, parent=args[0])
            else:
                show_error_dialog(error_message=ase)

    return wrapper


def show_error_dialog(error_message: Exception, parent: Optional[QWidget] = None) -> None:
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Critical)
    msg.setText(f"{type(error_message).__name__} occurred")
    msg.setInformativeText(str(error_message))
    msg.setWindowTitle("Error")
    msg.exec_()
