import sys
from window import MainWindow
from PyQt5.QtWidgets import QApplication
import parameters
from joystick import Joystick

def main():
    # Initialize components
    parameters.game_state = "Main_menu"
    
    # Start serial communication
    
    # Launch GUI
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    parameters.joystick_reader = Joystick()

    while(1):
        if parameters.game_state == "Main_menu":
            pass
        elif parameters.game_state == "Game":
            pass

if __name__ == "__main__":
    
    main()