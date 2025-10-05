from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QWidget, QScrollArea, QFrame, QComboBox)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QMouseEvent
import json
import os
import parameters

class KeyBindingDialog(QDialog):
    def __init__(self, parent=None, config_file="key_bindings.json"):
        super().__init__(parent)
        self.config_file = config_file
        self.waiting_for_key = None
        self.waiting_label = None
        
        # Zawsze resetuj key_bindings aby uniknąć referencji do usuniętych widgetów
        # z poprzednich instancji dialogu
        parameters.key_bindings = {}
        
        self.init_ui()
        self.load_config()
    
    def init_ui(self):
        self.setWindowTitle("Przypisanie klawiszy")
        self.setGeometry(150, 150, 600, 700)
        self.setModal(True)
        
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Title
        title = QLabel("Konfiguracja przycisków")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; padding: 15px;")
        main_layout.addWidget(title)
        
        # Scroll area for key bindings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        main_layout.addWidget(scroll)
        
        # Container for key binding items
        container = QWidget()
        self.bindings_layout = QVBoxLayout()
        container.setLayout(self.bindings_layout)
        scroll.setWidget(container)
        
        # Add key binding rows
        self.add_binding_row("Joystick Lewy", "joy_left", binding_type="dropdown")
        self.add_binding_row("Joystick Prawy", "joy_right", binding_type="dropdown")
        self.add_binding_row("Przycisk Lewy Joystick", "joy_lb", binding_type="key")
        self.add_binding_row("Przycisk Prawy Joystick", "joy_rb", binding_type="key")
        
        # Add some spacing
        self.bindings_layout.addStretch()
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        self.btn_save = QPushButton("Zapisz")
        self.btn_save.setFixedSize(150, 40)
        self.btn_save.clicked.connect(self.on_save)
        button_layout.addWidget(self.btn_save)
        
        self.btn_cancel = QPushButton("Anuluj")
        self.btn_cancel.setFixedSize(150, 40)
        self.btn_cancel.clicked.connect(self.on_cancel)
        button_layout.addWidget(self.btn_cancel)
        
        main_layout.addLayout(button_layout)
    
    def add_binding_row(self, label_text, key_id, binding_type="dropdown"):
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_widget.setLayout(row_layout)
        row_widget.setStyleSheet("padding: 5px;")
        
        # Label
        label = QLabel(label_text)
        label.setFixedWidth(250)
        label.setStyleSheet("font-size: 14px;")
        row_layout.addWidget(label)
        
        if binding_type == "dropdown":
            # Dropdown list for joystick actions
            combo = QComboBox()
            combo.setFixedWidth(250)
            combo.setStyleSheet("""
                QComboBox {
                    background-color: #f0f0f0;
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    padding: 5px;
                    font-size: 12px;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QComboBox::down-arrow {
                    image: none;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 5px solid #666;
                }
            """)
            
            # Add options
            combo.addItem("Nie przypisano", None)
            combo.addItem("Ruch kamery", "camera_move")
            combo.addItem("Ruch postaci WSAD", "character_move_wsad")
            combo.addItem("Ruch postaci STRZAŁKI", "character_move_arrows")
            
            row_layout.addWidget(combo)
            
            # Store reference in parameters
            parameters.key_bindings[key_id] = {
                'widget': combo,
                'type': 'joystick',
                'value': None
            }
        elif binding_type == "key":
            # Label showing current key assignment
            key_label = QLabel("Nie przypisano")
            key_label.setFixedWidth(150)
            key_label.setAlignment(Qt.AlignCenter)
            key_label.setStyleSheet("""
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 5px;
                font-size: 12px;
            """)
            row_layout.addWidget(key_label)
            
            # Button to assign key
            assign_btn = QPushButton("Przypisz klawisz")
            assign_btn.setFixedSize(150, 40)
            assign_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            assign_btn.clicked.connect(lambda: self.on_assign_button(key_id, key_label))
            row_layout.addWidget(assign_btn)
            
            # Store reference in parameters
            parameters.key_bindings[key_id] = {
                'widget': key_label,
                'button': assign_btn,
                'type': 'key',
                'value': None
            }
        
        row_layout.addStretch()
        self.bindings_layout.addWidget(row_widget)
    
    def on_assign_button(self, key_id, label):
        """Start waiting for key/mouse button assignment"""
        self.waiting_for_key = key_id
        self.waiting_label = label
        
        label.setText("Naciśnij klawisz lub przycisk myszy...")
        label.setStyleSheet("""
            background-color: #ffffcc;
            border: 2px solid #ff9900;
            border-radius: 3px;
            padding: 5px;
            font-size: 12px;
            font-weight: bold;
        """)
        
        # Enable mouse tracking to capture mouse clicks
        self.setMouseTracking(True)
        self.grabMouse()
        self.setFocus()
    
    def keyPressEvent(self, event):
        """Capture keyboard key press"""
        if self.waiting_for_key:
            # Get key name
            key = event.key()
            key_text = event.text()
            
            # Map special keys to readable names
            key_names = {
                Qt.Key_Space: "Space",
                Qt.Key_Return: "Enter",
                Qt.Key_Enter: "Enter",
                Qt.Key_Tab: "Tab",
                Qt.Key_Backspace: "Backspace",
                Qt.Key_Escape: "Escape",
                Qt.Key_Shift: "Shift",
                Qt.Key_Control: "Ctrl",
                Qt.Key_Alt: "Alt",
                Qt.Key_CapsLock: "Caps Lock",
                Qt.Key_Up: "↑",
                Qt.Key_Down: "↓",
                Qt.Key_Left: "←",
                Qt.Key_Right: "→",
            }
            
            if key in key_names:
                key_string = key_names[key]
            elif key_text and key_text.isprintable():
                key_string = key_text.upper()
            else:
                key_string = f"Key_{key}"
            
            self.assign_key(key_string)
        else:
            super().keyPressEvent(event)
    
    def mousePressEvent(self, event):
        """Capture mouse button press"""
        if self.waiting_for_key:
            button = event.button()
            
            # Map mouse buttons to readable names
            button_names = {
                Qt.LeftButton: "Lewy przycisk myszy",
                Qt.RightButton: "Prawy przycisk myszy",
                Qt.MiddleButton: "Środkowy przycisk myszy",
            }
            
            if button in button_names:
                self.assign_key(button_names[button])
            else:
                self.assign_key(f"Mouse_{button}")
        else:
            super().mousePressEvent(event)
    
    def assign_key(self, key_string):
        """Assign the captured key/button"""
        if self.waiting_for_key and self.waiting_label:
            # Update display
            self.waiting_label.setText(key_string)
            self.waiting_label.setStyleSheet("""
                background-color: #ccffcc;
                border: 2px solid #00cc00;
                border-radius: 3px;
                padding: 5px;
                font-size: 12px;
            """)
            
            # Store binding in parameters
            parameters.key_bindings[self.waiting_for_key]['value'] = key_string
            
            # Clean up
            self.waiting_for_key = None
            self.waiting_label = None
            self.releaseMouse()
            self.setMouseTracking(False)
    
    def save_config(self):
        """Save current configuration to JSON file"""
        config_data = {}
        
        for key_id, data in parameters.key_bindings.items():
            if data['type'] == 'joystick':
                # For dropdown, get selected value
                combo = data['widget']
                config_data[key_id] = combo.currentData()
            elif data['type'] == 'key':
                # For key binding, get stored value
                config_data[key_id] = data['value']
        
        print(f"Konfiguracja w parameters.key_bindings:")
        print(config_data)
        
        # Zapisz do pliku JSON
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
            print(f"Konfiguracja zapisana do pliku: {self.config_file}")
            return True
        except Exception as e:
            print(f"Błąd podczas zapisu konfiguracji do pliku: {e}")
            return False
    
    def load_config(self):
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_file):
            print(f"Plik konfiguracyjny nie istnieje: {self.config_file}")
            return False
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            print(f"Konfiguracja wczytana z pliku: {self.config_file}")
        except Exception as e:
            print(f"Błąd podczas wczytywania konfiguracji: {e}")
            return False
        
        # Zastosuj wczytaną konfigurację
        for key_id, value in config_data.items():
            if key_id not in parameters.key_bindings:
                continue
            
            binding = parameters.key_bindings[key_id]
            
            if binding['type'] == 'joystick':
                # Set dropdown value
                combo = binding['widget']
                index = combo.findData(value)
                if index >= 0:
                    combo.setCurrentIndex(index)
                binding['value'] = value
                
            elif binding['type'] == 'key':
                # Set key binding value
                if value:
                    binding['widget'].setText(value)
                    binding['widget'].setStyleSheet("""
                        background-color: #ccffcc;
                        border: 2px solid #00cc00;
                        border-radius: 3px;
                        padding: 5px;
                        font-size: 12px;
                    """)
                    binding['value'] = value
        
        return True
    
    def on_save(self):
        """Save key bindings and close"""
        # Update values from widgets before saving
        for key_id, data in parameters.key_bindings.items():
            if data['type'] == 'joystick':
                combo = data['widget']
                data['value'] = combo.currentData()
        
        print("Zapisane przypisania:")
        for key_id, data in parameters.key_bindings.items():
            print(f"  {key_id}: {data['value']}")
        
        # Save to JSON file
        if self.save_config():
            self.accept()
        else:
            print("Nie udało się zapisać konfiguracji, ale dialog zostanie zamknięty")
            self.accept()
    
    def on_cancel(self):
        """Cancel and close without saving"""
        self.reject()
    
    def get_bindings(self):
        """Return current key bindings"""
        result = {}
        for key_id, data in parameters.key_bindings.items():
            if data['type'] == 'joystick':
                combo = data['widget']
                result[key_id] = combo.currentData()
            else:
                result[key_id] = data['value']
        return result