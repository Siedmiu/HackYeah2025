import pyautogui
import parameters

class Joystick():
    def __init__(self):
        # Joystick range 0-4095 (12-bit ADC)
        self.min_value = 0
        self.max_value = 4095
        self.center = 2048         
        
        # Zwiększona strefa martwa - reaguj tylko na wyraźne wychylenie
        self.deadzone_radius = 600  # Zwiększono z 300 na 600
        self.deadzone_min = self.center - self.deadzone_radius
        self.deadzone_max = self.center + self.deadzone_radius
        
        # Prędkość ruchu
        self.movement_speed = 10
        self.camera_sensitivity = 10  # Zwiększono czułość kamery
        
        self.prev_left_button = 0
        self.prev_right_button = 0
        
        # WSAD and arrows mapping
        self.wsad_keys = {'up': 'a', 'down': 'd', 'left': 's', 'right': 'w'}
        self.arrow_keys = {'up': 'left', 'down': 'right', 'left': 'down', 'right': 'up'}
        
        self.pressed_keys = set()

    def update(self):
        if parameters.game_state != "Game":
            self.release_all_keys()
            return
        
        left_x = parameters.joystick_left_x
        left_y = parameters.joystick_left_y
        left_btn = parameters.joystick_left_button
        right_x = parameters.joystick_right_x
        right_y = parameters.joystick_right_y
        right_btn = parameters.joystick_right_button
        
        # Pobierz przypisania klawiszy z konfiguracji
        left_btn_binding = parameters.key_bindings.get('joy_lb', {}).get('value')
        right_btn_binding = parameters.key_bindings.get('joy_rb', {}).get('value')
        
        # STAŁE PRZYPISANIE: Lewy joystick = ruch myszki, Prawy joystick = WSAD
        self.handle_camera_movement(left_x, left_y)
        self.handle_character_movement(right_x, right_y, self.wsad_keys)
        
        # Obsługa przycisków joysticków według konfiguracji
        self.handle_button(left_btn, self.prev_left_button, left_btn_binding)
        self.handle_button(right_btn, self.prev_right_button, right_btn_binding)
        
        self.prev_left_button = left_btn
        self.prev_right_button = right_btn

    def handle_character_movement(self, x, y, key_map):
        keys_to_press = set()
        
        # Określ które klawisze powinny być wciśnięte
        if x < self.deadzone_min:
            keys_to_press.add(key_map['left'])
        elif x > self.deadzone_max:
            keys_to_press.add(key_map['right'])
        
        if y < self.deadzone_min:
            keys_to_press.add(key_map['up'])
        elif y > self.deadzone_max:
            keys_to_press.add(key_map['down'])
        
        # POPRAWKA: Zwalniaj tylko te klawisze z tej mapy, które są obecnie wciśnięte
        # ale nie powinny być
        keys_from_this_map = set(key_map.values())
        keys_to_release = (self.pressed_keys & keys_from_this_map) - keys_to_press
        
        for key in keys_to_release:
            pyautogui.keyUp(key)
            self.pressed_keys.discard(key)
        
        # Wciskaj nowe klawisze
        keys_to_press_new = keys_to_press - self.pressed_keys
        for key in keys_to_press_new:
            pyautogui.keyDown(key)
            self.pressed_keys.add(key)

    def handle_camera_movement(self, x, y):
        offset_x = 0
        offset_y = 0
        
        if x < self.deadzone_min:
            normalized = (x - self.deadzone_min) / (self.deadzone_min - self.min_value)
            offset_x = normalized * self.camera_sensitivity
        elif x > self.deadzone_max:
            normalized = (x - self.deadzone_max) / (self.max_value - self.deadzone_max)
            offset_x = normalized * self.camera_sensitivity
        
        if y < self.deadzone_min:
            normalized = (y - self.deadzone_min) / (self.deadzone_min - self.min_value)
            offset_y = normalized * self.camera_sensitivity
        elif y > self.deadzone_max:
            normalized = (y - self.deadzone_max) / (self.max_value - self.deadzone_max)
            offset_y = normalized * self.camera_sensitivity
        
        if offset_x != 0 or offset_y != 0:
            pyautogui.moveRel(int(offset_x), int(offset_y))

    def handle_button(self, current_state, previous_state, key_binding):
        if key_binding is None:
            return
        
        if current_state == 1 and previous_state == 0:
            self.press_key(key_binding)
        elif current_state == 0 and previous_state == 1:
            self.release_key(key_binding)

    def press_key(self, key_binding):
        if key_binding == "Lewy przycisk myszy":
            pyautogui.mouseDown(button='left')
        elif key_binding == "Prawy przycisk myszy":
            pyautogui.mouseDown(button='right')
        elif key_binding == "Środkowy przycisk myszy":
            pyautogui.mouseDown(button='middle')
        elif key_binding == "Space":
            pyautogui.keyDown('space')
        else:
            pyautogui.keyDown(key_binding.lower())

    def release_key(self, key_binding):
        if key_binding == "Lewy przycisk myszy":
            pyautogui.mouseUp(button='left')
        elif key_binding == "Prawy przycisk myszy":
            pyautogui.mouseUp(button='right')
        elif key_binding == "Środkowy przycisk myszy":
            pyautogui.mouseUp(button='middle')
        elif key_binding == "Space":
            pyautogui.keyUp('space')
        else:
            pyautogui.keyUp(key_binding.lower())

    def release_all_keys(self):
        for key in self.pressed_keys.copy():
            pyautogui.keyUp(key)
        self.pressed_keys.clear()