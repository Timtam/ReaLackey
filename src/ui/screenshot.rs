//! Window / screen capture for the Phase 7 vision tools.
//!
//! Captures pixels via GDI `BitBlt` from the screen DC (which renders
//! GPU-accelerated Direct2D/OpenGL/DXGI plugin GUIs correctly, unlike
//! `PrintWindow`, which draws them black), converts BGRA→RGBA, optionally
//! downscales, PNG-encodes, and base64s the result for the Anthropic image
//! content block. Windows-only; other platforms return an error so callers
//! degrade gracefully.
//!
//! Downscaling is applied only to the description-only targets (the REAPER main
//! window and the full screen), which can exceed the vision API's size limit.
//! The focused-plugin capture is left 1:1 so Tier-B pixel coordinates map
//! exactly to the window. All calls run on the REAPER main thread.
//!
//! macOS (Core Graphics / ScreenCaptureKit) is a dedicated later pass.

/// A captured image, ready for an Anthropic image block.
pub struct Shot {
    /// Base64 of the PNG bytes.
    pub png_base64: String,
    pub width: u32,
    pub height: u32,
}

/// Capture the window `hwnd`. If `bring_to_front`, un-minimize/raise it first (so
/// it isn't obscured). If `max_edge` is set, downscale so the long edge fits.
#[cfg(windows)]
pub fn capture_hwnd(hwnd: isize, bring_to_front: bool, max_edge: Option<u32>) -> Result<Shot, String> {
    imp::capture_hwnd(hwnd, bring_to_front, max_edge)
}

/// Capture the whole virtual desktop (all monitors), optionally downscaled.
#[cfg(windows)]
pub fn capture_screen(max_edge: Option<u32>) -> Result<Shot, String> {
    imp::capture_screen(max_edge)
}

#[cfg(not(windows))]
pub fn capture_hwnd(_hwnd: isize, _front: bool, _max_edge: Option<u32>) -> Result<Shot, String> {
    Err("screen capture is not implemented on this platform yet (macOS backend pending)".into())
}

#[cfg(not(windows))]
pub fn capture_screen(_max_edge: Option<u32>) -> Result<Shot, String> {
    Err("screen capture is not implemented on this platform yet (macOS backend pending)".into())
}

#[cfg(windows)]
mod imp {
    use super::Shot;
    use base64::Engine;
    use windows::Win32::Foundation::{HWND, RECT};
    use windows::Win32::Graphics::Gdi::{
        BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC,
        GetDIBits, ReleaseDC, SelectObject, UpdateWindow, BITMAPINFO, BITMAPINFOHEADER,
        DIB_RGB_COLORS, HBITMAP, HDC, HGDIOBJ, SRCCOPY,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        BringWindowToTop, GetSystemMetrics, GetWindowRect, IsIconic, SetForegroundWindow,
        ShowWindow, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
        SW_RESTORE,
    };

    // GDI's BI_RGB (uncompressed), as the plain u32 biCompression wants.
    const BI_RGB_U32: u32 = 0;
    const MAX_BYTES: usize = 512 * 1024 * 1024;

    struct ScreenDc(HDC);
    impl Drop for ScreenDc {
        fn drop(&mut self) {
            unsafe {
                ReleaseDC(None, self.0);
            }
        }
    }
    struct MemDc(HDC);
    impl Drop for MemDc {
        fn drop(&mut self) {
            unsafe {
                let _ = DeleteDC(self.0);
            }
        }
    }
    struct Obj(HGDIOBJ);
    impl Drop for Obj {
        fn drop(&mut self) {
            unsafe {
                let _ = DeleteObject(self.0);
            }
        }
    }

    pub fn capture_hwnd(
        hwnd_raw: isize,
        bring_to_front: bool,
        max_edge: Option<u32>,
    ) -> Result<Shot, String> {
        let hwnd = HWND(hwnd_raw as *mut core::ffi::c_void);
        if hwnd.0.is_null() {
            return Err("null window handle".into());
        }
        unsafe {
            if bring_to_front {
                if IsIconic(hwnd).as_bool() {
                    let _ = ShowWindow(hwnd, SW_RESTORE);
                }
                let _ = SetForegroundWindow(hwnd);
                let _ = BringWindowToTop(hwnd);
            }
            // Force a synchronous repaint (a just-floated window may not have
            // painted yet). No-op for already-painted windows.
            let _ = UpdateWindow(hwnd);

            let mut rect = RECT::default();
            GetWindowRect(hwnd, &mut rect).map_err(|e| format!("GetWindowRect failed: {e}"))?;
            let w = rect.right - rect.left;
            let h = rect.bottom - rect.top;
            if w <= 0 || h <= 0 {
                return Err(format!("window has non-positive size ({w}x{h})"));
            }
            capture_rect(rect.left, rect.top, w, h, max_edge)
        }
    }

    pub fn capture_screen(max_edge: Option<u32>) -> Result<Shot, String> {
        // SAFETY: GetSystemMetrics has no preconditions; capture_rect is safe to
        // call on the main thread.
        unsafe {
            let x = GetSystemMetrics(SM_XVIRTUALSCREEN);
            let y = GetSystemMetrics(SM_YVIRTUALSCREEN);
            let w = GetSystemMetrics(SM_CXVIRTUALSCREEN);
            let h = GetSystemMetrics(SM_CYVIRTUALSCREEN);
            if w <= 0 || h <= 0 {
                return Err("virtual screen has no area".into());
            }
            capture_rect(x, y, w, h, max_edge)
        }
    }

    /// BitBlt a screen rectangle into an RGBA buffer, optionally downscale it,
    /// then PNG-encode and base64. `(x, y)` are screen (virtual-desktop) coords.
    unsafe fn capture_rect(
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        max_edge: Option<u32>,
    ) -> Result<Shot, String> {
        let screen_dc = GetDC(None);
        if screen_dc.is_invalid() {
            return Err("GetDC(screen) failed".into());
        }
        let _screen = ScreenDc(screen_dc);

        let mem_dc = CreateCompatibleDC(Some(screen_dc));
        if mem_dc.is_invalid() {
            return Err("CreateCompatibleDC failed".into());
        }
        let _mem = MemDc(mem_dc);

        let bitmap: HBITMAP = CreateCompatibleBitmap(screen_dc, w, h);
        if bitmap.is_invalid() {
            return Err("CreateCompatibleBitmap failed".into());
        }
        let _bmp = Obj(HGDIOBJ(bitmap.0));

        // Select, blit, restore — GetDIBits needs the bitmap NOT selected. Validate
        // the select so a failure can't leave the bitmap selected.
        let old = SelectObject(mem_dc, HGDIOBJ(bitmap.0));
        if old.0.is_null() {
            return Err("SelectObject (select bitmap) failed".into());
        }
        let blt = BitBlt(mem_dc, 0, 0, w, h, Some(screen_dc), x, y, SRCCOPY);
        SelectObject(mem_dc, old);
        blt.map_err(|e| format!("BitBlt failed: {e}"))?;

        let mut info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: core::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: w,
                biHeight: -h, // top-down
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB_U32,
                ..Default::default()
            },
            ..Default::default()
        };
        let buf_size = (w as usize)
            .checked_mul(h as usize)
            .and_then(|n| n.checked_mul(4))
            .filter(|&n| n <= MAX_BYTES)
            .ok_or_else(|| format!("capture area too large ({w}x{h})"))?;
        let mut buf = vec![0u8; buf_size];
        let scanlines = GetDIBits(
            mem_dc,
            bitmap,
            0,
            h as u32,
            Some(buf.as_mut_ptr() as *mut core::ffi::c_void),
            &mut info,
            DIB_RGB_COLORS,
        );
        if scanlines == 0 {
            return Err("GetDIBits returned no scanlines".into());
        }

        // BGRA → RGBA, force opaque alpha (window alpha is unreliable).
        for px in buf.chunks_exact_mut(4) {
            px.swap(0, 2);
            px[3] = 255;
        }

        let (out_buf, out_w, out_h) = match max_edge {
            Some(m) if (w as u32).max(h as u32) > m && m > 0 => {
                downscale(&buf, w as u32, h as u32, m)
            }
            _ => (buf, w as u32, h as u32),
        };

        let png = encode_png(out_w, out_h, &out_buf)?;
        let png_base64 = base64::engine::general_purpose::STANDARD.encode(&png);
        Ok(Shot {
            png_base64,
            width: out_w,
            height: out_h,
        })
    }

    /// Integer box-average downscale so the long edge is <= `max_edge`. Used only
    /// for description-only captures (main window / full screen).
    fn downscale(rgba: &[u8], w: u32, h: u32, max_edge: u32) -> (Vec<u8>, u32, u32) {
        let factor = w.max(h).div_ceil(max_edge).max(1);
        if factor <= 1 {
            return (rgba.to_vec(), w, h);
        }
        let tw = (w / factor).max(1);
        let th = (h / factor).max(1);
        let mut out = vec![0u8; (tw as usize) * (th as usize) * 4];
        for ty in 0..th {
            for tx in 0..tw {
                let (mut r, mut g, mut b, mut a, mut count) = (0u32, 0u32, 0u32, 0u32, 0u32);
                for dy in 0..factor {
                    let sy = ty * factor + dy;
                    if sy >= h {
                        break;
                    }
                    for dx in 0..factor {
                        let sx = tx * factor + dx;
                        if sx >= w {
                            break;
                        }
                        let i = ((sy * w + sx) * 4) as usize;
                        r += rgba[i] as u32;
                        g += rgba[i + 1] as u32;
                        b += rgba[i + 2] as u32;
                        a += rgba[i + 3] as u32;
                        count += 1;
                    }
                }
                let count = count.max(1);
                let o = ((ty * tw + tx) * 4) as usize;
                out[o] = (r / count) as u8;
                out[o + 1] = (g / count) as u8;
                out[o + 2] = (b / count) as u8;
                out[o + 3] = (a / count) as u8;
            }
        }
        (out, tw, th)
    }

    fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, width, height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder
                .write_header()
                .map_err(|e| format!("PNG header: {e}"))?;
            writer
                .write_image_data(rgba)
                .map_err(|e| format!("PNG data: {e}"))?;
        }
        Ok(out)
    }
}
