import sys
from window import MainWindow
from PyQt5.QtWidgets import QApplication

def main():
    # Initialize components
    
    # Start serial communication
    
    # Launch GUI
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    pass

if __name__ == "__main__":
    
    main()