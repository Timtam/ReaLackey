//! Synthetic mouse input into a window, for the Phase 7 Tier-B "operate a
//! GUI-only control" capability (e.g. a Kontakt mode toggle that has no host
//! parameter).
//!
//! Coordinates are in the pixel space of the screenshot the model saw (which is
//! the window's `GetWindowRect`, so image-(0,0) == the window's top-left). They
//! are hard-CLAMPED to the window's current bounds here, so a synthesized click
//! can never land outside the target plugin window.
//!
//! On Windows we use `SendInput` at real (absolute, virtual-desktop-normalized)
//! coordinates — the only thing custom-rendered GUIs (JUCE/Kontakt/OpenGL)
//! reliably respond to; posted `WM_*` messages are silently ignored by them. The
//! window is brought to the foreground (and un-minimized) FIRST so its rect is
//! current when we measure it, and the real cursor is moved and restored. All
//! calls run on the REAPER main thread.
//!
//! macOS (CGEvent) is a dedicated later pass; non-Windows returns an error.

/// Single left click at image-space `(x, y)` within window `hwnd`.
#[cfg(windows)]
pub fn click(hwnd: isize, x: i32, y: i32) -> Result<(), String> {
    imp::click(hwnd, x, y)
}

/// Double left click at image-space `(x, y)` (select a field, reset a knob…).
#[cfg(windows)]
pub fn double_click(hwnd: isize, x: i32, y: i32) -> Result<(), String> {
    imp::double_click(hwnd, x, y)
}

/// Left-button drag from `(x1, y1)` to `(x2, y2)` in image space (knob turns).
#[cfg(windows)]
pub fn drag(hwnd: isize, x1: i32, y1: i32, x2: i32, y2: i32) -> Result<(), String> {
    imp::drag(hwnd, x1, y1, x2, y2)
}

/// Type Unicode `text` into whatever control in `hwnd` has focus (e.g. a value
/// field just clicked); if `submit`, press Enter afterward.
#[cfg(windows)]
pub fn type_text(hwnd: isize, text: &str, submit: bool) -> Result<(), String> {
    imp::type_text(hwnd, text, submit)
}

/// Mouse-wheel scroll by `clicks` notches (positive = up/away) at image-space
/// `(x, y)` within `hwnd`.
#[cfg(windows)]
pub fn scroll(hwnd: isize, x: i32, y: i32, clicks: i32) -> Result<(), String> {
    imp::scroll(hwnd, x, y, clicks)
}

#[cfg(not(windows))]
const UNSUPPORTED: &str =
    "synthetic input is not implemented on this platform yet (macOS backend pending)";

#[cfg(not(windows))]
pub fn click(_hwnd: isize, _x: i32, _y: i32) -> Result<(), String> {
    Err(UNSUPPORTED.into())
}

#[cfg(not(windows))]
pub fn double_click(_hwnd: isize, _x: i32, _y: i32) -> Result<(), String> {
    Err(UNSUPPORTED.into())
}

#[cfg(not(windows))]
pub fn drag(_hwnd: isize, _x1: i32, _y1: i32, _x2: i32, _y2: i32) -> Result<(), String> {
    Err(UNSUPPORTED.into())
}

#[cfg(not(windows))]
pub fn type_text(_hwnd: isize, _text: &str, _submit: bool) -> Result<(), String> {
    Err(UNSUPPORTED.into())
}

#[cfg(not(windows))]
pub fn scroll(_hwnd: isize, _x: i32, _y: i32, _clicks: i32) -> Result<(), String> {
    Err(UNSUPPORTED.into())
}

#[cfg(windows)]
mod imp {
    use windows::Win32::Foundation::{HWND, POINT, RECT};
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT, KEYBD_EVENT_FLAGS,
        KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_LEFTDOWN,
        MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MOVE, MOUSEEVENTF_VIRTUALDESK, MOUSEEVENTF_WHEEL,
        MOUSEINPUT, MOUSE_EVENT_FLAGS, VIRTUAL_KEY, VK_RETURN,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        BringWindowToTop, GetCursorPos, GetSystemMetrics, GetWindowRect, IsIconic, SetCursorPos,
        SetForegroundWindow, ShowWindow, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN,
        SM_YVIRTUALSCREEN, SW_RESTORE,
    };

    /// Interpolation points for a drag (enough for smooth knob tracking).
    const DRAG_STEPS: i32 = 24;

    pub fn click(hwnd_raw: isize, x: i32, y: i32) -> Result<(), String> {
        let hwnd = validate(hwnd_raw)?;
        // SAFETY: main thread; hwnd validated non-null.
        unsafe {
            // Focus/raise/un-minimize FIRST, then measure the rect, so the
            // coordinate mapping reflects the window's actual on-screen position.
            let restore = focus_and_save(hwnd);
            let (sx, sy) = clamp_to_window(hwnd, x, y)?;
            let (nx, ny) = normalize(sx, sy);
            send(&[
                mouse(nx, ny, MOUSEEVENTF_MOVE),
                mouse(nx, ny, MOUSEEVENTF_LEFTDOWN),
                mouse(nx, ny, MOUSEEVENTF_LEFTUP),
            ])?;
            restore_cursor(restore);
        }
        Ok(())
    }

    pub fn double_click(hwnd_raw: isize, x: i32, y: i32) -> Result<(), String> {
        let hwnd = validate(hwnd_raw)?;
        // SAFETY: main thread; hwnd validated non-null.
        unsafe {
            let restore = focus_and_save(hwnd);
            let (sx, sy) = clamp_to_window(hwnd, x, y)?;
            let (nx, ny) = normalize(sx, sy);
            // Two down/up pairs in one batch land within the double-click time.
            send(&[
                mouse(nx, ny, MOUSEEVENTF_MOVE),
                mouse(nx, ny, MOUSEEVENTF_LEFTDOWN),
                mouse(nx, ny, MOUSEEVENTF_LEFTUP),
                mouse(nx, ny, MOUSEEVENTF_LEFTDOWN),
                mouse(nx, ny, MOUSEEVENTF_LEFTUP),
            ])?;
            restore_cursor(restore);
        }
        Ok(())
    }

    pub fn type_text(hwnd_raw: isize, text: &str, submit: bool) -> Result<(), String> {
        let hwnd = validate(hwnd_raw)?;
        // SAFETY: main thread; hwnd validated non-null.
        unsafe {
            // Foreground so keystrokes reach the plugin's focused control.
            let _ = focus_and_save(hwnd);
            let mut inputs = Vec::new();
            for unit in text.encode_utf16() {
                inputs.push(key_unicode(unit, false));
                inputs.push(key_unicode(unit, true));
            }
            if submit {
                inputs.push(key_vk(VK_RETURN, false));
                inputs.push(key_vk(VK_RETURN, true));
            }
            if !inputs.is_empty() {
                send(&inputs)?;
            }
        }
        Ok(())
    }

    pub fn scroll(hwnd_raw: isize, x: i32, y: i32, clicks: i32) -> Result<(), String> {
        let hwnd = validate(hwnd_raw)?;
        // SAFETY: main thread; hwnd validated non-null.
        unsafe {
            let restore = focus_and_save(hwnd);
            let (sx, sy) = clamp_to_window(hwnd, x, y)?;
            let (nx, ny) = normalize(sx, sy);
            let delta = clicks.saturating_mul(120); // WHEEL_DELTA per notch
            send(&[mouse(nx, ny, MOUSEEVENTF_MOVE), wheel(nx, ny, delta)])?;
            restore_cursor(restore);
        }
        Ok(())
    }

    pub fn drag(hwnd_raw: isize, x1: i32, y1: i32, x2: i32, y2: i32) -> Result<(), String> {
        let hwnd = validate(hwnd_raw)?;
        // SAFETY: main thread; hwnd validated non-null.
        unsafe {
            let restore = focus_and_save(hwnd);
            let (sx1, sy1) = clamp_to_window(hwnd, x1, y1)?;
            let (sx2, sy2) = clamp_to_window(hwnd, x2, y2)?;
            let mut inputs = Vec::with_capacity(DRAG_STEPS as usize + 3);
            let (fnx, fny) = normalize(sx1, sy1);
            inputs.push(mouse(fnx, fny, MOUSEEVENTF_MOVE));
            inputs.push(mouse(fnx, fny, MOUSEEVENTF_LEFTDOWN));
            for i in 1..=DRAG_STEPS {
                let t = i as f64 / DRAG_STEPS as f64;
                let ix = sx1 + ((sx2 - sx1) as f64 * t).round() as i32;
                let iy = sy1 + ((sy2 - sy1) as f64 * t).round() as i32;
                let (nx, ny) = normalize(ix, iy);
                inputs.push(mouse(nx, ny, MOUSEEVENTF_MOVE));
            }
            let (lnx, lny) = normalize(sx2, sy2);
            inputs.push(mouse(lnx, lny, MOUSEEVENTF_LEFTUP));
            send(&inputs)?;
            restore_cursor(restore);
        }
        Ok(())
    }

    /// Parse and null-check the raw handle.
    fn validate(hwnd_raw: isize) -> Result<HWND, String> {
        let hwnd = HWND(hwnd_raw as *mut core::ffi::c_void);
        if hwnd.0.is_null() {
            return Err("null window handle".into());
        }
        Ok(hwnd)
    }

    /// Read the window's CURRENT rect, clamp `(x, y)` into it, and return the
    /// absolute screen pixel of the clamped point (so a click can't leave it).
    /// Call this AFTER `focus_and_save` so the rect is up to date.
    unsafe fn clamp_to_window(hwnd: HWND, x: i32, y: i32) -> Result<(i32, i32), String> {
        let mut rect = RECT::default();
        GetWindowRect(hwnd, &mut rect).map_err(|e| format!("GetWindowRect: {e}"))?;
        let w = rect.right - rect.left;
        let h = rect.bottom - rect.top;
        if w <= 0 || h <= 0 {
            return Err("target window has no area".into());
        }
        let cx = x.clamp(0, w - 1);
        let cy = y.clamp(0, h - 1);
        Ok((rect.left + cx, rect.top + cy))
    }

    /// Un-minimize, foreground, and raise the window so it receives the input and
    /// is where we measure it; save the cursor position so we can put it back.
    unsafe fn focus_and_save(hwnd: HWND) -> Option<POINT> {
        if IsIconic(hwnd).as_bool() {
            let _ = ShowWindow(hwnd, SW_RESTORE);
        }
        let _ = SetForegroundWindow(hwnd);
        let _ = BringWindowToTop(hwnd);
        let mut p = POINT::default();
        if GetCursorPos(&mut p).is_ok() {
            Some(p)
        } else {
            None
        }
    }

    unsafe fn restore_cursor(saved: Option<POINT>) {
        if let Some(p) = saved {
            let _ = SetCursorPos(p.x, p.y);
        }
    }

    /// Map an absolute screen pixel to the 0..65535 range SendInput expects for
    /// MOUSEEVENTF_ABSOLUTE over the whole virtual desktop (multi-monitor safe).
    /// Clamped, so an off-screen window rect can't produce out-of-range values.
    fn normalize(screen_x: i32, screen_y: i32) -> (i32, i32) {
        // SAFETY: GetSystemMetrics has no preconditions.
        let (vx, vy, vw, vh) = unsafe {
            (
                GetSystemMetrics(SM_XVIRTUALSCREEN),
                GetSystemMetrics(SM_YVIRTUALSCREEN),
                GetSystemMetrics(SM_CXVIRTUALSCREEN),
                GetSystemMetrics(SM_CYVIRTUALSCREEN),
            )
        };
        (norm(screen_x - vx, vw), norm(screen_y - vy, vh))
    }

    fn norm(offset: i32, size: i32) -> i32 {
        let size = size.max(1) as i64;
        (((offset as i64) * 65535 + size / 2) / size).clamp(0, 65535) as i32
    }

    fn mouse(nx: i32, ny: i32, flags: MOUSE_EVENT_FLAGS) -> INPUT {
        INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: nx,
                    dy: ny,
                    mouseData: 0,
                    dwFlags: MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK | flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    fn wheel(nx: i32, ny: i32, delta: i32) -> INPUT {
        INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: nx,
                    dy: ny,
                    // Signed wheel delta carried as its two's-complement u32.
                    mouseData: delta as u32,
                    dwFlags: MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK | MOUSEEVENTF_WHEEL,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    /// A Unicode character as a keyboard event (layout-independent typing).
    fn key_unicode(unit: u16, up: bool) -> INPUT {
        let mut flags = KEYEVENTF_UNICODE;
        if up {
            flags |= KEYEVENTF_KEYUP;
        }
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: VIRTUAL_KEY(0),
                    wScan: unit,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    /// A virtual-key event (e.g. Enter to submit a value).
    fn key_vk(vk: VIRTUAL_KEY, up: bool) -> INPUT {
        let flags = if up {
            KEYEVENTF_KEYUP
        } else {
            KEYBD_EVENT_FLAGS(0)
        };
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: vk,
                    wScan: 0,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    unsafe fn send(inputs: &[INPUT]) -> Result<(), String> {
        let sent = SendInput(inputs, core::mem::size_of::<INPUT>() as i32);
        if sent as usize == inputs.len() {
            Ok(())
        } else {
            Err(format!(
                "SendInput injected {sent} of {} events (input blocked?)",
                inputs.len()
            ))
        }
    }
}
