//! Small text utilities for presentation.
//!
//! The model replies in Markdown. A screen reader would otherwise speak the
//! markup itself ("hash", "star star", "backtick"), so `strip_markdown` renders
//! the content down to clean prose for OSARA announcements. The same parser will
//! later feed the HTML output pane.

use pulldown_cmark::{Event, Parser, TagEnd};

/// Convert Markdown to plain, readable prose (no `#`, `*`, backticks, link URLs,
/// list bullets). Soft line-wraps become spaces so sentences flow when spoken.
pub fn strip_markdown(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for event in Parser::new(input) {
        match event {
            Event::Text(t) | Event::Code(t) => out.push_str(&t),
            Event::SoftBreak => out.push(' '),
            Event::HardBreak | Event::Rule => out.push('\n'),
            Event::End(
                TagEnd::Paragraph
                | TagEnd::Heading(_)
                | TagEnd::Item
                | TagEnd::CodeBlock
                | TagEnd::BlockQuote(_),
            ) => out.push('\n'),
            _ => {}
        }
    }
    collapse_blank_lines(out.trim())
}

/// Collapse runs of 3+ newlines down to a paragraph break.
fn collapse_blank_lines(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut newlines = 0;
    for ch in s.chars() {
        if ch == '\n' {
            newlines += 1;
            if newlines <= 2 {
                result.push('\n');
            }
        } else {
            newlines = 0;
            result.push(ch);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_headings_emphasis_and_code() {
        let s = strip_markdown("# Title\n\nThis is **bold**, *italic*, and `code`.");
        assert!(!s.contains('#'), "{s}");
        assert!(!s.contains('*'), "{s}");
        assert!(!s.contains('`'), "{s}");
        assert!(s.contains("Title") && s.contains("bold") && s.contains("italic") && s.contains("code"));
    }

    #[test]
    fn links_become_their_text() {
        let s = strip_markdown("See [the docs](https://example.com/x) for more.");
        assert!(s.contains("the docs"), "{s}");
        assert!(!s.contains("https://"), "{s}");
        assert!(!s.contains('[') && !s.contains(']'), "{s}");
    }

    #[test]
    fn list_markers_are_removed() {
        let s = strip_markdown("- one\n- two\n- three");
        assert!(!s.contains("- "), "{s}");
        assert!(s.contains("one") && s.contains("two") && s.contains("three"));
    }

    #[test]
    fn code_block_content_survives_without_fences() {
        let s = strip_markdown("Here:\n\n```rust\nlet x = 1;\n```\n");
        assert!(!s.contains("```"), "{s}");
        assert!(s.contains("let x = 1;"), "{s}");
    }

    #[test]
    fn plain_text_is_unchanged_modulo_trim() {
        assert_eq!(strip_markdown("Just a sentence."), "Just a sentence.");
    }
}
