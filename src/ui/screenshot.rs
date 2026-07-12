//! Window / screen capture for the Phase 7 vision tools.
//!
//! Windows: GDI `BitBlt` from the screen DC (which renders GPU-accelerated
//! Direct2D/OpenGL/DXGI plugin GUIs correctly, unlike `PrintWindow`, which draws
//! them black). macOS: Core Graphics captures the screen region at the window's
//! rect (window raised first) — the same "grab the on-screen pixels" strategy,
//! so it handles GPU GUIs too. Both convert to RGBA, optionally downscale,
//! PNG-encode and base64. Other platforms return an error so callers degrade.
//!
//! Downscaling is applied only to the description-only targets (the REAPER main
//! window and the full screen), which can exceed the vision API's size limit.
//! The focused-plugin capture is left 1:1 so Tier-B pixel coordinates map to the
//! window. All calls run on the REAPER main thread.
//!
//! NOTE: the macOS backend is UNVALIDATED until a macOS CI build compiles it (we
//! develop on Windows). Known follow-up: HiDPI/Retina — a points→pixels scale
//! factor between the captured image and the window rect affects Tier-B input
//! coordinate mapping.

/// A captured image, ready for an Anthropic image block.
pub struct Shot {
    /// Base64 of the PNG bytes.
    pub png_base64: String,
    pub width: u32,
    pub height: u32,
}

/// Capture the window `hwnd`. If `bring_to_front`, un-minimize/raise it first (so
/// it isn't obscured). If `max_edge` is set, downscale so the long edge fits.
pub fn capture_hwnd(
    hwnd: isize,
    bring_to_front: bool,
    max_edge: Option<u32>,
) -> Result<Shot, String> {
    imp::capture_hwnd(hwnd, bring_to_front, max_edge)
}

/// Capture the whole desktop (all monitors), optionally downscaled.
pub fn capture_screen(max_edge: Option<u32>) -> Result<Shot, String> {
    imp::capture_screen(max_edge)
}

// ---- shared post-processing (RGBA -> optional downscale -> PNG -> base64) ----

/// Take a top-down RGBA buffer, optionally downscale so the long edge fits
/// `max_edge`, then PNG-encode and base64. Shared by every platform backend.
#[cfg(any(windows, target_os = "macos"))]
fn finish_rgba(rgba: Vec<u8>, w: u32, h: u32, max_edge: Option<u32>) -> Result<Shot, String> {
    use base64::Engine;
    let (out_buf, out_w, out_h) = match max_edge {
        Some(m) if w.max(h) > m && m > 0 => downscale(&rgba, w, h, m),
        _ => (rgba, w, h),
    };
    let png = encode_png(out_w, out_h, &out_buf)?;
    let png_base64 = base64::engine::general_purpose::STANDARD.encode(&png);
    Ok(Shot {
        png_base64,
        width: out_w,
        height: out_h,
    })
}

/// Integer box-average downscale so the long edge is <= `max_edge`.
#[cfg(any(windows, target_os = "macos"))]
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

#[cfg(any(windows, target_os = "macos"))]
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

// ---- Windows: GDI BitBlt from the screen DC ---------------------------------
#[cfg(windows)]
mod imp {
    use super::Shot;
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

    /// BitBlt a screen rectangle into a top-down RGBA buffer, then hand off to the
    /// shared post-processing. `(x, y)` are screen (virtual-desktop) coords.
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

        // BGRA -> RGBA, force opaque alpha (window alpha is unreliable).
        for px in buf.chunks_exact_mut(4) {
            px.swap(0, 2);
            px[3] = 255;
        }
        super::finish_rgba(buf, w as u32, h as u32, max_edge)
    }
}

// ---- macOS: Core Graphics capture of the on-screen window rect --------------
// UNVALIDATED: compiled only on a macOS build. Uses the SWELL C-shim for window
// geometry + raise (cross-platform), and Core Graphics for the actual capture.
#[cfg(target_os = "macos")]
mod imp {
    use super::Shot;
    use crate::ui::ffi;
    use core_graphics::display::CGDisplay;
    use core_graphics::geometry::{CGPoint, CGRect, CGSize};
    use core_graphics::window::{kCGNullWindowID, kCGWindowListOptionOnScreenOnly};

    pub fn capture_hwnd(
        hwnd: isize,
        bring_to_front: bool,
        max_edge: Option<u32>,
    ) -> Result<Shot, String> {
        if hwnd == 0 {
            return Err("null window handle".into());
        }
        if bring_to_front {
            ffi::window_to_front(hwnd);
        }
        let (x, y, w, h) = ffi::window_rect(hwnd).ok_or("could not read the window rect")?;
        if w <= 0 || h <= 0 {
            return Err(format!("window has non-positive size ({w}x{h})"));
        }
        let rect = CGRect::new(
            &CGPoint::new(x as f64, y as f64),
            &CGSize::new(w as f64, h as f64),
        );
        let image = CGDisplay::screenshot(
            rect,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
            core_graphics::window::kCGWindowImageDefault,
        )
        .ok_or("CGDisplay screenshot returned no image")?;
        cgimage_to_shot(&image, max_edge)
    }

    pub fn capture_screen(max_edge: Option<u32>) -> Result<Shot, String> {
        let image = CGDisplay::main()
            .image()
            .ok_or("CGDisplay image returned nothing")?;
        cgimage_to_shot(&image, max_edge)
    }

    /// Convert a CGImage (32-bit, little-endian BGRA-in-memory) to a tight RGBA
    /// buffer (dropping any per-row padding), then hand to shared post-processing.
    fn cgimage_to_shot(
        image: &core_graphics::image::CGImage,
        max_edge: Option<u32>,
    ) -> Result<Shot, String> {
        let w = image.width();
        let h = image.height();
        let bytes_per_row = image.bytes_per_row();
        let data = image.data();
        let src = data.bytes();
        if w == 0 || h == 0 || bytes_per_row < w * 4 {
            return Err(format!("unexpected CGImage geometry ({w}x{h}, stride {bytes_per_row})"));
        }
        let mut rgba = vec![0u8; w * h * 4];
        for row in 0..h {
            let s = row * bytes_per_row;
            let d = row * w * 4;
            for col in 0..w {
                let sp = s + col * 4;
                let dp = d + col * 4;
                // memory order is BGRA -> RGBA; force opaque alpha.
                rgba[dp] = src[sp + 2];
                rgba[dp + 1] = src[sp + 1];
                rgba[dp + 2] = src[sp];
                rgba[dp + 3] = 255;
            }
        }
        super::finish_rgba(rgba, w as u32, h as u32, max_edge)
    }
}

// ---- other platforms: not implemented ---------------------------------------
#[cfg(all(not(windows), not(target_os = "macos")))]
mod imp {
    use super::Shot;
    pub fn capture_hwnd(_h: isize, _f: bool, _m: Option<u32>) -> Result<Shot, String> {
        Err("screen capture is not implemented on this platform".into())
    }
    pub fn capture_screen(_m: Option<u32>) -> Result<Shot, String> {
        Err("screen capture is not implemented on this platform".into())
    }
}
