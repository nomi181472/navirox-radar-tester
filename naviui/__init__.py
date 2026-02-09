"""
NaviUI - Autonomous Navigation System Dashboard
A sophisticated PyQt6 dark-themed maritime command interface

Author: Shaikh Azan Asim
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Shaikh Azan Asim"

from .app import MainWindow
from .styles import DARK_STYLESHEET

__all__ = ["MainWindow", "DARK_STYLESHEET", "__version__"]
