#pragma once

using KeyState = u8;
enum : KeyState {
	KeyState_none       = 0x0,
	KeyState_held       = 0x1,
	KeyState_down       = 0x2,
	KeyState_up         = 0x4,
	KeyState_repeated   = 0x8,
	KeyState_clicked    = 0x80,
};
::KeyState key_state[256 + 3];

bool key_down  (u8 key) { return key_state[key] & KeyState_down    ; }
bool key_up    (u8 key) { return key_state[key] & KeyState_up      ; }
bool key_repeat(u8 key) { return key_state[key] & KeyState_repeated; }
bool key_held  (u8 key) { return key_state[key] & KeyState_held    ; }

bool mouse_down  (u8 key) { return key_state[256+key] & KeyState_down    ; }
bool mouse_up    (u8 key) { return key_state[256+key] & KeyState_up      ; }
bool mouse_repeat(u8 key) { return key_state[256+key] & KeyState_repeated; }
bool mouse_held  (u8 key) { return key_state[256+key] & KeyState_held    ; }

v2f mouse_delta;
f32 cursor_speed;

f32 mouse_wheel_delta;

v2u window_client_size;
