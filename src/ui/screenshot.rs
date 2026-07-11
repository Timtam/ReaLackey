//! Window screen-capture for the Phase 7 vision tools.
//!
//! Captures a window's pixels via GDI `BitBlt` from the screen DC (which renders
//! GPU-accelerated Direct2D/OpenGL/DXGI plugin GUIs correctly, unlike
//! `PrintWindow`, which draws them black), converts BGRA→RGBA, PNG-encodes, and
//! base64s the result for the Anthropic image content block. Windows-only; other
//! platforms return an error so callers degrade gracefully.
//!
//! All calls run on the REAPER main thread (GDI is not thread-safe for a foreign
//! window's DC). A robustness pass (black-buffer detection + `PrintWindow`
//! fallback + DPI/multi-monitor handling) lands in a later milestone.

/// A captured window image, ready for an Anthropic image block.
pub struct Shot {
    /// Base64 of the PNG bytes.
    pub png_base64: String,
    pub width: u32,
    pub height: u32,
}

/// Capture the window identified by `hwnd` (a raw `HWND` as an `isize`).
#[cfg(windows)]
pub fn capture_hwnd(hwnd: isize) -> Result<Shot, String> {
    imp::capture_hwnd(hwnd)
}

#[cfg(not(windows))]
pub fn capture_hwnd(_hwnd: isize) -> Result<Shot, String> {
    // The macOS backend (Core Graphics / ScreenCaptureKit, mapping the SWELL
    // HWND -> NSWindow -> CGWindowID, gated by Screen-Recording permission)
    // lands in the dedicated macOS pass. Until then, no capture off-Windows.
    Err("screen capture is not implemented on this platform yet (macOS backend pending)".into())
}

#[cfg(windows)]
mod imp {
    use super::Shot;
    use base64::Engine;
    use windows::Win32::Foundation::{HWND, RECT};
    use windows::Win32::Graphics::Gdi::{
        BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC,
        GetDIBits, ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, DIB_RGB_COLORS, HBITMAP,
        HDC, HGDIOBJ, SRCCOPY,
    };
    use windows::Win32::UI::WindowsAndMessaging::GetWindowRect;

    // GDI's BI_RGB (uncompressed) compression constant, as the plain u32 the
    // BITMAPINFOHEADER.biCompression field wants.
    const BI_RGB_U32: u32 = 0;

    /// Releases a screen DC obtained via `GetDC(None)` on drop.
    struct ScreenDc(HDC);
    impl Drop for ScreenDc {
        fn drop(&mut self) {
            unsafe {
                ReleaseDC(None, self.0);
            }
        }
    }

    /// Deletes a memory DC created via `CreateCompatibleDC` on drop.
    struct MemDc(HDC);
    impl Drop for MemDc {
        fn drop(&mut self) {
            unsafe {
                let _ = DeleteDC(self.0);
            }
        }
    }

    /// Deletes a GDI object (our bitmap) on drop.
    struct Obj(HGDIOBJ);
    impl Drop for Obj {
        fn drop(&mut self) {
            unsafe {
                let _ = DeleteObject(self.0);
            }
        }
    }

    pub fn capture_hwnd(hwnd_raw: isize) -> Result<Shot, String> {
        let hwnd = HWND(hwnd_raw as *mut core::ffi::c_void);
        if hwnd.0.is_null() {
            return Err("null window handle".into());
        }

        unsafe {
            let mut rect = RECT::default();
            GetWindowRect(hwnd, &mut rect).map_err(|e| format!("GetWindowRect failed: {e}"))?;
            let width = rect.right - rect.left;
            let height = rect.bottom - rect.top;
            if width <= 0 || height <= 0 {
                return Err(format!("window has non-positive size ({width}x{height})"));
            }

            // Screen (desktop) DC; we blit the window's screen rectangle out of it
            // so GPU-composited content is captured (PrintWindow would be black).
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

            let bitmap: HBITMAP = CreateCompatibleBitmap(screen_dc, width, height);
            if bitmap.is_invalid() {
                return Err("CreateCompatibleBitmap failed".into());
            }
            let _bmp = Obj(HGDIOBJ(bitmap.0));

            // Select the bitmap, blit, then restore — GetDIBits requires the
            // bitmap NOT be selected into a DC. Validate the select: a null
            // return means it failed, in which case restoring would leave the
            // bitmap selected and corrupt the later GetDIBits.
            let old = SelectObject(mem_dc, HGDIOBJ(bitmap.0));
            if old.0.is_null() {
                return Err("SelectObject (select bitmap) failed".into());
            }
            let blt = BitBlt(
                mem_dc,
                0,
                0,
                width,
                height,
                Some(screen_dc),
                rect.left,
                rect.top,
                SRCCOPY,
            );
            SelectObject(mem_dc, old); // restore the DC's original bitmap
            blt.map_err(|e| format!("BitBlt failed: {e}"))?;

            // Pull the pixels as top-down 32-bit BGRA (negative height = top-down).
            let mut info = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: core::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: width,
                    biHeight: -height,
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB_U32,
                    ..Default::default()
                },
                ..Default::default()
            };
            // Checked so an exotic/huge window rect can't overflow the size
            // computation (which would under-allocate and let GetDIBits write
            // out of bounds); also cap it to a sane ceiling to avoid a wild OOM.
            const MAX_BYTES: usize = 512 * 1024 * 1024;
            let buf_size = (width as usize)
                .checked_mul(height as usize)
                .and_then(|n| n.checked_mul(4))
                .filter(|&n| n <= MAX_BYTES)
                .ok_or_else(|| format!("window too large to capture ({width}x{height})"))?;
            let mut buf = vec![0u8; buf_size];
            let scanlines = GetDIBits(
                mem_dc,
                bitmap,
                0,
                height as u32,
                Some(buf.as_mut_ptr() as *mut core::ffi::c_void),
                &mut info,
                DIB_RGB_COLORS,
            );
            if scanlines == 0 {
                return Err("GetDIBits returned no scanlines".into());
            }

            // BGRA → RGBA, and force opaque alpha (window alpha is unreliable).
            for px in buf.chunks_exact_mut(4) {
                px.swap(0, 2);
                px[3] = 255;
            }

            let png = encode_png(width as u32, height as u32, &buf)?;
            let png_base64 = base64::engine::general_purpose::STANDARD.encode(&png);
            Ok(Shot {
                png_base64,
                width: width as u32,
                height: height as u32,
            })
        }
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
