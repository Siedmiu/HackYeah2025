import pyautogui
import parameters

class Joystick():
    def __init__(self):
        # Threshold dla joysticka (zakres 0-4096)
        self.center = 2048  # Środek zakresu
        
        # Strefa martwa (deadzone) wokół centrum
        self.deadzone_radius = 300  # +/- 300 od centrum = strefa martwa
        self.deadzone_min = self.center - self.deadzone_radius  # 1748
        self.deadzone_max = self.center + self.deadzone_radius  # 2348
        
        # Pełny zakres dla maksymalnego wychylenia
        self.min_value = 0
        self.max_value = 4096
        
        # Prędkość ruchu (piksele lub wartość dla pyautogui)
        self.movement_speed = 10
        self.camera_sensitivity = 5
        
        # Stan poprzednich przycisków (do wykrywania kliknięć)
        self.prev_left_button = 0
        self.prev_right_button = 0
        
        # Mapowanie klawiszy dla WSAD i strzałek
        self.wsad_keys = {'up': 's', 'down': 'w', 'left': 'a', 'right': 'd'}
        self.arrow_keys = {'up': 'left', 'down': 'right', 'left': 'down', 'right': 'up'}
        
        # Aktualnie wciśnięte klawisze (do śledzenia)
        self.pressed_keys = set()

    def set_keys(self):
        """Aktualizacja ustawień klawiszy z parameters.key_bindings"""
        # Ta metoda może być używana do dynamicznego odświeżania bindingów
        pass

    def update(self):
        """Główna metoda aktualizacji - wywoływana w każdej klatce"""
        if parameters.game_state != "Game":
            self.release_all_keys()
            return
        
        # Pobierz aktualne wartości joysticka z parameters
        left_x = parameters.joystick_left_x
        left_y = parameters.joystick_left_y
        left_btn = parameters.joystick_left_button
        right_x = parameters.joystick_right_x
        right_y = parameters.joystick_right_y
        right_btn = parameters.joystick_right_button
        
        # Sprawdź bindingi z parameters.key_bindings
        left_binding = parameters.key_bindings.get('joy_left', {}).get('value')
        right_binding = parameters.key_bindings.get('joy_right', {}).get('value')
        left_btn_binding = parameters.key_bindings.get('joy_lb', {}).get('value')
        right_btn_binding = parameters.key_bindings.get('joy_rb', {}).get('value')
        
        # Obsługa lewego joysticka
        if left_binding == "character_move_wsad":
            self.handle_character_movement(left_x, left_y, self.wsad_keys)
        elif left_binding == "character_move_arrows":
            self.handle_character_movement(left_x, left_y, self.arrow_keys)
        elif left_binding == "camera_move":
            self.handle_camera_movement(left_x, left_y)
        
        # Obsługa prawego joysticka
        if right_binding == "character_move_wsad":
            self.handle_character_movement(right_x, right_y, self.wsad_keys)
        elif right_binding == "character_move_arrows":
            self.handle_character_movement(right_x, right_y, self.arrow_keys)
        elif right_binding == "camera_move":
            self.handle_camera_movement(right_x, right_y)
        
        # Obsługa przycisków joysticka
        self.handle_button(left_btn, self.prev_left_button, left_btn_binding)
        self.handle_button(right_btn, self.prev_right_button, right_btn_binding)
        
        # Zapisz poprzedni stan przycisków
        self.prev_left_button = left_btn
        self.prev_right_button = right_btn

    def handle_character_movement(self, x, y, key_map):
        """Obsługa ruchu postaci (WSAD lub strzałki)"""
        keys_to_press = set()
        
        # Ruch w poziomie (X)
        if x < self.deadzone_min:
            keys_to_press.add(key_map['left'])
        elif x > self.deadzone_max:
            keys_to_press.add(key_map['right'])
        
        # Ruch w pionie (Y) - uwaga: joysticki często mają odwróconą oś Y
        if y < self.deadzone_min:
            keys_to_press.add(key_map['up'])
        elif y > self.deadzone_max:
            keys_to_press.add(key_map['down'])
        
        # Zwolnij klawisze, które nie są już używane
        keys_to_release = self.pressed_keys - keys_to_press
        for key in keys_to_release:
            if key in key_map.values():
                pyautogui.keyUp(key)
                self.pressed_keys.discard(key)
        
        # Wciśnij nowe klawisze
        keys_to_press_new = keys_to_press - self.pressed_keys
        for key in keys_to_press_new:
            pyautogui.keyDown(key)
            self.pressed_keys.add(key)

    def handle_camera_movement(self, x, y):
        """Obsługa ruchu kamery (ruch myszką)"""
        offset_x = 0
        offset_y = 0
        
        # Oblicz offset od centrum z uwzględnieniem deadzone
        if x < self.deadzone_min:
            # Ruch w lewo: normalizuj do zakresu -1.0 do 0.0
            normalized = (x - self.deadzone_min) / (self.deadzone_min - self.min_value)
            offset_x = normalized * self.camera_sensitivity
        elif x > self.deadzone_max:
            # Ruch w prawo: normalizuj do zakresu 0.0 do 1.0
            normalized = (x - self.deadzone_max) / (self.max_value - self.deadzone_max)
            offset_x = normalized * self.camera_sensitivity
        
        if y < self.deadzone_min:
            # Ruch w górę: normalizuj do zakresu -1.0 do 0.0
            normalized = (y - self.deadzone_min) / (self.deadzone_min - self.min_value)
            offset_y = normalized * self.camera_sensitivity
        elif y > self.deadzone_max:
            # Ruch w dół: normalizuj do zakresu 0.0 do 1.0
            normalized = (y - self.deadzone_max) / (self.max_value - self.deadzone_max)
            offset_y = normalized * self.camera_sensitivity
        
        # Przesuń mysz względnie
        if offset_x != 0 or offset_y != 0:
            pyautogui.moveRel(int(offset_x), int(offset_y))

    def handle_button(self, current_state, previous_state, key_binding):
        """Obsługa przycisku joysticka"""
        if key_binding is None:
            return
        
        # Wykryj naciśnięcie (zmiana z 0 na 1)
        if current_state == 1 and previous_state == 0:
            self.press_key(key_binding)
        # Wykryj puszczenie (zmiana z 1 na 0)
        elif current_state == 0 and previous_state == 1:
            self.release_key(key_binding)

    def press_key(self, key_binding):
        """Naciśnij klawisz lub przycisk myszy"""
        if key_binding == "Lewy przycisk myszy":
            pyautogui.mouseDown(button='left')
        elif key_binding == "Prawy przycisk myszy":
            pyautogui.mouseDown(button='right')
        elif key_binding == "Środkowy przycisk myszy":
            pyautogui.mouseDown(button='middle')
        elif key_binding == "Space":
            pyautogui.keyDown('space')
        else:
            # Zwykły klawisz
            pyautogui.keyDown(key_binding.lower())

    def release_key(self, key_binding):
        """Puść klawisz lub przycisk myszy"""
        if key_binding == "Lewy przycisk myszy":
            pyautogui.mouseUp(button='left')
        elif key_binding == "Prawy przycisk myszy":
            pyautogui.mouseUp(button='right')
        elif key_binding == "Środkowy przycisk myszy":
            pyautogui.mouseUp(button='middle')
        elif key_binding == "Space":
            pyautogui.keyUp('space')
        else:
            # Zwykły klawisz
            pyautogui.keyUp(key_binding.lower())

    def release_all_keys(self):
        """Zwolnij wszystkie wciśnięte klawisze (wywoływane przy wyjściu z trybu Game)"""
        for key in self.pressed_keys.copy():
            pyautogui.keyUp(key)
        self.pressed_keys.clear()