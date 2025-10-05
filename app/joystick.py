# joystick_controls.py
# Wersja z keyboard (klawiatura) + mouse (mysz)
# pip install keyboard mouse

import keyboard
import mouse
import parameters

class Joystick():
    def __init__(self):
        # Zakres joysticka 0-4096
        self.center = 2048

        # Dead zone
        self.deadzone_radius = 300
        self.deadzone_min = self.center - self.deadzone_radius
        self.deadzone_max = self.center + self.deadzone_radius

        # Pełny zakres
        self.min_value = 0
        self.max_value = 4096

        # czułość kamery (piksele na tick)
        self.camera_sensitivity = 5

        self.prev_left_button = 0
        self.prev_right_button = 0

        # Mapowanie WSAD i strzałek
        self.wsad_keys = {'up': 'w', 'down': 's', 'left': 'a', 'right': 'd'}
        # Jeśli chcesz normalne strzałki, zostaw tak:
        self.arrow_keys = {'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right'}

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

        left_binding = parameters.key_bindings.get('joy_left', {}).get('value')
        right_binding = parameters.key_bindings.get('joy_right', {}).get('value')
        left_btn_binding = parameters.key_bindings.get('joy_lb', {}).get('value')
        right_btn_binding = parameters.key_bindings.get('joy_rb', {}).get('value')

        if left_binding == "character_move_wsad":
            self.handle_character_movement(left_x, left_y, self.wsad_keys)
        elif left_binding == "character_move_arrows":
            self.handle_character_movement(left_x, left_y, self.arrow_keys)
        elif left_binding == "camera_move":
            self.handle_camera_movement(left_x, left_y)

        if right_binding == "character_move_wsad":
            self.handle_character_movement(right_x, right_y, self.wsad_keys)
        elif right_binding == "character_move_arrows":
            self.handle_character_movement(right_x, right_y, self.arrow_keys)
        elif right_binding == "camera_move":
            self.handle_camera_movement(right_x, right_y)

        self.handle_button(left_btn, self.prev_left_button, left_btn_binding)
        self.handle_button(right_btn, self.prev_right_button, right_btn_binding)

        self.prev_left_button = left_btn
        self.prev_right_button = right_btn

    def handle_character_movement(self, x, y, key_map):
        keys_to_press = set()

        if x < self.deadzone_min:
            keys_to_press.add(key_map['left'])
        elif x > self.deadzone_max:
            keys_to_press.add(key_map['right'])

        if y < self.deadzone_min:
            keys_to_press.add(key_map['up'])
        elif y > self.deadzone_max:
            keys_to_press.add(key_map['down'])

        # Zwolnij te, których już nie trzeba trzymać
        keys_to_release = self.pressed_keys - keys_to_press
        for key in list(keys_to_release):
            if key in key_map.values():
                keyboard.release(key)
                self.pressed_keys.discard(key)

        # Wciśnij nowe
        keys_to_press_new = keys_to_press - self.pressed_keys
        for key in keys_to_press_new:
            keyboard.press(key)
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

        # Uwaga: w wielu grach oś Y jest "odwrócona".
        if y < self.deadzone_min:
            normalized = (y - self.deadzone_min) / (self.deadzone_min - self.min_value)
            offset_y = normalized * self.camera_sensitivity
        elif y > self.deadzone_max:
            normalized = (y - self.deadzone_max) / (self.max_value - self.deadzone_max)
            offset_y = normalized * self.camera_sensitivity

        dx = int(offset_x)
        dy = int(offset_y)
        if dx != 0 or dy != 0:
            # ruch względny (bez animacji)
            mouse.move(dx, dy, absolute=False, duration=0)

    def handle_button(self, current_state, previous_state, key_binding):
        if key_binding is None:
            return

        if current_state == 1 and previous_state == 0:
            self.press_key(key_binding)
        elif current_state == 0 and previous_state == 1:
            self.release_key(key_binding)

    def press_key(self, key_binding):
        if key_binding == "Lewy przycisk myszy":
            mouse.press(button='left')
        elif key_binding == "Prawy przycisk myszy":
            mouse.press(button='right')
        elif key_binding == "Środkowy przycisk myszy":
            mouse.press(button='middle')
        elif key_binding.lower() == "space":
            keyboard.press('space')
            self.pressed_keys.add('space')
        else:
            k = key_binding.lower()
            keyboard.press(k)
            self.pressed_keys.add(k)

    def release_key(self, key_binding):
        if key_binding == "Lewy przycisk myszy":
            mouse.release(button='left')
        elif key_binding == "Prawy przycisk myszy":
            mouse.release(button='right')
        elif key_binding == "Środkowy przycisk myszy":
            mouse.release(button='middle')
        elif key_binding.lower() == "space":
            keyboard.release('space')
            self.pressed_keys.discard('space')
        else:
            k = key_binding.lower()
            keyboard.release(k)
            self.pressed_keys.discard(k)

    def release_all_keys(self):
        for key in list(self.pressed_keys):
            keyboard.release(key)
        self.pressed_keys.clear()
