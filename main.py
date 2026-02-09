"""
NaviUI - Autonomous Navigation System Dashboard
A sophisticated PyQt6 dark-themed maritime command interface.

Author: Shaikh Azan Asim
Version: 1.0.0

This is the application entry point. The main application logic 
has been modularized into the naviui package.
"""

import sys
from PyQt6.QtWidgets import QApplication

from naviui import MainWindow, DARK_STYLESHEET


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style as base
    app.setStyleSheet(DARK_STYLESHEET)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
