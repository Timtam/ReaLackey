//! Small text utilities for presentation.
//!
//! The model replies in Markdown. A screen reader would otherwise speak the
//! markup itself ("hash", "star star", "backtick"), so `strip_markdown` renders
//! the content down to clean prose for OSARA announcements. The same parser will
//! later feed the HTML output pane.

use pulldown_cmark::{html, CowStr, Event, LinkType, Options, Parser, Tag, TagEnd};

/// Render Markdown to HTML for the webview output pane. Raw HTML embedded in the
/// model's text is neutralized — rendered as escaped, visible text rather than
/// live DOM — because the result is injected via `innerHTML`, so a reflected
/// `<img onerror=…>`/`<iframe>` (e.g. from prompt-injected tool output) must not
/// execute. pulldown-cmark already escapes ordinary text/code.
///
/// Bare `http(s)://…` URLs in ordinary text are auto-linked (pulldown-cmark only
/// links explicit `[text](url)` and `<url>` autolinks). URLs inside code spans/
/// blocks or already inside a link are left untouched.
pub fn markdown_to_html(input: &str) -> String {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TABLES);
    // Depth counters: don't auto-link inside an existing link (would nest <a>) or
    // inside a code block (URLs there are literal).
    let mut in_link = 0usize;
    let mut in_code = 0usize;
    let parser = Parser::new_ext(input, options).flat_map(move |event| {
        match &event {
            Event::Start(Tag::Link { .. }) => in_link += 1,
            Event::End(TagEnd::Link) => in_link = in_link.saturating_sub(1),
            Event::Start(Tag::CodeBlock(_)) => in_code += 1,
            Event::End(TagEnd::CodeBlock) => in_code = in_code.saturating_sub(1),
            _ => {}
        }
        match event {
            // Turn raw HTML into plain text so push_html escapes it (never linkified).
            Event::Html(h) | Event::InlineHtml(h) => vec![Event::Text(h)],
            Event::Text(t) if in_link == 0 && in_code == 0 => linkify(&t),
            other => vec![other],
        }
    });
    let mut out = String::with_capacity(input.len() + input.len() / 2);
    html::push_html(&mut out, parser);
    out
}

/// Split a run of text into Text + autolink events for each bare `http(s)://` URL.
/// Returns a single Text event when there is no URL.
fn linkify<'a>(text: &str) -> Vec<Event<'a>> {
    let mut out: Vec<Event<'a>> = Vec::new();
    let mut rest = text;
    loop {
        match find_scheme(rest) {
            None => {
                if !rest.is_empty() {
                    out.push(Event::Text(rest.to_string().into()));
                }
                break;
            }
            Some(pos) => {
                let end = pos + url_len(&rest[pos..]);
                if pos > 0 {
                    out.push(Event::Text(rest[..pos].to_string().into()));
                }
                let url = &rest[pos..end];
                out.push(Event::Start(Tag::Link {
                    link_type: LinkType::Autolink,
                    dest_url: url.to_string().into(),
                    title: CowStr::from(""),
                    id: CowStr::from(""),
                }));
                out.push(Event::Text(url.to_string().into()));
                out.push(Event::End(TagEnd::Link));
                rest = &rest[end..];
            }
        }
    }
    if out.is_empty() {
        out.push(Event::Text(text.to_string().into()));
    }
    out
}

/// Byte offset of the next `http://`/`https://` that begins at a word boundary
/// (so `shttp://…` inside a token isn't matched), or None.
fn find_scheme(s: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(rel) = s[from..].find("http") {
        let idx = from + rel;
        let tail = &s[idx..];
        let boundary = idx == 0 || !s.as_bytes()[idx - 1].is_ascii_alphanumeric();
        if boundary && (tail.starts_with("http://") || tail.starts_with("https://")) {
            return Some(idx);
        }
        from = idx + 4;
        if from >= s.len() {
            break;
        }
    }
    None
}

/// Length of the URL starting at `s` (which begins with the scheme): everything
/// up to whitespace or a delimiter, minus trailing punctuation that is usually
/// sentence/wrapping punctuation rather than part of the URL.
fn url_len(s: &str) -> usize {
    let mut end = 0;
    for (i, c) in s.char_indices() {
        if c.is_whitespace() || matches!(c, '<' | '>' | '"' | '`' | '\'' | '|') {
            break;
        }
        end = i + c.len_utf8();
    }
    let b = s.as_bytes();
    while end > 0 {
        match b[end - 1] {
            b'.' | b',' | b';' | b':' | b'!' | b'?' => end -= 1,
            // Trim a trailing ')' or ']' only if it is unbalanced (e.g. the URL
            // sits inside parentheses: "(see https://x)").
            b')' if s[..end].matches('(').count() < s[..end].matches(')').count() => end -= 1,
            b']' if s[..end].matches('[').count() < s[..end].matches(']').count() => end -= 1,
            _ => break,
        }
    }
    end
}

/// Escape a plain string for safe inclusion in HTML text/attribute context.
pub fn html_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

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
        assert!(
            s.contains("Title") && s.contains("bold") && s.contains("italic") && s.contains("code")
        );
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

    #[test]
    fn markdown_to_html_renders_and_escapes() {
        let h = markdown_to_html("**bold** and `co<de>`");
        assert!(h.contains("<strong>bold</strong>"), "{h}");
        assert!(h.contains("<code>"), "{h}");
        // The `<de>` inside code must be escaped, not treated as a tag.
        assert!(h.contains("co&lt;de&gt;"), "{h}");
    }

    #[test]
    fn markdown_to_html_neutralizes_raw_html() {
        // Reflected raw HTML (e.g. from prompt-injected tool output) must be
        // rendered as inert, escaped text — never live DOM.
        let h = markdown_to_html("hi <img src=x onerror=alert(1)> <iframe></iframe>");
        assert!(!h.contains("<img"), "{h}");
        assert!(!h.contains("<iframe"), "{h}");
        assert!(h.contains("&lt;img"), "{h}");
    }

    #[test]
    fn html_escape_neutralizes_markup() {
        assert_eq!(html_escape("a<b>&\"'"), "a&lt;b&gt;&amp;&quot;&#39;");
    }

    #[test]
    fn bare_url_is_autolinked() {
        let h = markdown_to_html("see https://example.com/x now");
        assert!(
            h.contains("<a href=\"https://example.com/x\">https://example.com/x</a>"),
            "{h}"
        );
        assert!(h.contains("see ") && h.contains(" now"), "{h}");
    }

    #[test]
    fn trailing_punctuation_is_not_part_of_the_url() {
        let h = markdown_to_html("docs at https://example.com/a.");
        assert!(h.contains("href=\"https://example.com/a\""), "{h}");
        // the sentence period stays outside the link
        assert!(h.contains("</a>."), "{h}");
    }

    #[test]
    fn url_wrapped_in_parens_excludes_the_closing_paren() {
        let h = markdown_to_html("(https://example.com)");
        assert!(h.contains("href=\"https://example.com\""), "{h}");
        assert!(h.contains("</a>)"), "{h}");
    }

    #[test]
    fn urls_in_code_are_not_autolinked() {
        let inline = markdown_to_html("run `curl https://example.com`");
        assert!(!inline.contains("<a "), "inline code must not autolink: {inline}");
        let block = markdown_to_html("```\nhttps://example.com\n```");
        assert!(!block.contains("<a "), "code block must not autolink: {block}");
    }

    #[test]
    fn existing_markdown_link_is_not_double_wrapped() {
        let h = markdown_to_html("[the docs](https://example.com)");
        // exactly one anchor, with the label as its text (not the URL twice)
        assert_eq!(h.matches("<a ").count(), 1, "{h}");
        assert!(h.contains(">the docs</a>"), "{h}");
    }
}
