use crate::tts::normalize;
use crate::tts::vocab::VOCAB;
use lazy_static::lazy_static;
use regex::Regex;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use misaki_rs::G2P;

lazy_static! {
    // Pattern to insert space before "hˈʌndɹɪd" when it follows certain characters
    // Simplified from lookbehind: match [a-zɹː] followed by hˈʌndɹɪd and replace with space
    static ref PHONEME_PATTERNS: Regex = Regex::new(r"([a-zɹː])(hˈʌndɹɪd)").unwrap();
    // Pattern to fix " z" before punctuation or end of string
    // Simplified: match " z" followed by punctuation or end of string
    static ref Z_PATTERN: Regex = Regex::new(r#" z([;:,.!?¡¿—…"«»"" ]|$)"#).unwrap();
    // Pattern to fix "ninti" -> "nindi" (simplified from lookbehind/lookahead)
    // Match "nˈaɪnti" not followed by "ː"
    static ref NINETY_PATTERN: Regex = Regex::new(r"(nˈaɪn)ti([^ː]|$)").unwrap();
}

/// Backend type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Misaki (Rule-based) phonemizer backend
    Misaki,
}

/// Misaki backend implementation (uses misaki-rs)
struct MisakiBackend {
    g2p: G2P,
}

// Global phonemizer singleton - only used by new_auto() for backward compatibility
// The main new() method doesn't use singleton (like TTS OrtKoko)
static GLOBAL_PHONEMIZER: OnceLock<Mutex<Option<Arc<Phonemizer>>>> = OnceLock::new();

impl MisakiBackend {
    fn new(lang: &str) -> Result<Self, String> {
        let language = if lang == "b" || lang == "en-gb" {
            misaki_rs::Language::EnglishGB
        } else {
            misaki_rs::Language::EnglishUS
        };
        Ok(Self {
            g2p: G2P::new(language),
        })
    }

    fn phonemize(&self, text: &str, _language: &str) -> Option<String> {
        let (phonemes, _) = self.g2p.g2p(text);
        Some(phonemes)
    }
}

/// Enum to hold different backend implementations
enum Backend {
    Misaki(MisakiBackend),
}

impl Backend {
    fn phonemize(&self, text: &str, language: &str) -> Option<String> {
        match self {
            Backend::Misaki(backend) => backend.phonemize(text, language),
        }
    }

    fn set_model_path(&mut self, _path: PathBuf) {
        // ONNX backend uses the path passed during initialization
        tracing::debug!("set_model_path called on ONNX backend (no-op)");
    }
}

pub struct Phonemizer {
    lang: String,
    backend: Backend,
    backend_type: BackendType,
    model_path: Option<PathBuf>,
}

impl Phonemizer {
    /// Create a new Phonemizer with Misaki backend
    pub fn new(lang: &str, _model_path: &str) -> Result<Self, String> {
        tracing::info!("Initializing Misaki G2P phonemizer (rule-based)");
        
        // Create phonemizer
        let phonemizer = Self::new_with_backend_result(lang, BackendType::Misaki, None)?;
        
        tracing::info!("✓ G2P phonemizer initialized successfully");
        Ok(phonemizer)
    }

    /// Create a new Phonemizer with automatic path detection (for backward compatibility)
    pub fn new_auto(lang: &str) -> Result<Arc<Self>, String> {
        // Get or initialize the global phonemizer singleton
        let guard = GLOBAL_PHONEMIZER.get_or_init(|| Mutex::new(None));
        let mut phonemizer_opt = guard.lock().map_err(|e| {
            format!(
                "Failed to acquire global phonemizer lock (mutex poisoned): {:?}",
                e
            )
        })?;

        // If already initialized, return existing Arc
        if let Some(ref existing) = *phonemizer_opt {
            tracing::debug!("Reusing existing global phonemizer instance");
            return Ok(Arc::clone(existing));
        }

        // Initialize new phonemizer
        tracing::info!("Initializing global phonemizer (Misaki)");
        
        let phonemizer = Self::new_with_backend_result(lang, BackendType::Misaki, None)?;
        let phonemizer_arc = Arc::new(phonemizer);

        // Store in global singleton
        *phonemizer_opt = Some(Arc::clone(&phonemizer_arc));
        tracing::info!("✓ Global phonemizer singleton initialized and cached");

        Ok(phonemizer_arc)
    }

    pub fn new_or_panic(lang: &str, model_path: &str) -> Self {
        Self::new(lang, model_path).unwrap_or_else(|e| {
            tracing::error!("Failed to create Phonemizer: {}", e);
            panic!("Failed to create Phonemizer: {}", e);
        })
    }

    pub fn new_with_backend_result(
        lang: &str,
        backend_type: BackendType,
        _model_path: Option<PathBuf>,
    ) -> Result<Self, String> {
        let backend = match backend_type {
            BackendType::Misaki => Backend::Misaki(MisakiBackend::new(lang)?),
        };

        Ok(Phonemizer {
            lang: lang.to_string(),
            backend,
            backend_type,
            model_path: _model_path,
        })
    }

    /// Create a new Phonemizer with specified backend (panics on error, for backward compatibility)
    pub fn new_with_backend(
        lang: &str,
        backend_type: BackendType,
        model_path: Option<PathBuf>,
    ) -> Self {
        Self::new_with_backend_result(lang, backend_type, model_path)
            .unwrap_or_else(|e| panic!("Failed to create Phonemizer: {}", e))
    }

    /// Get the current backend type
    pub fn backend_type(&self) -> BackendType {
        self.backend_type
    }

    /// Set model path (no-op for ONNX, set during initialization)
    pub fn set_model_path(&mut self, path: PathBuf) {
        self.model_path = Some(path.clone());
        self.backend.set_model_path(path);
    }

    pub fn phonemize(&self, text: &str, normalize: bool) -> String {
        // Handle empty string case early - return empty result without processing
        if text.trim().is_empty() {
            return String::new();
        }

        // Wrap normalization in catch_unwind to prevent panics from crashing the app
        let text = if normalize {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                normalize::normalize_text(text)
            }))
            .unwrap_or_else(|_| {
                tracing::error!("Panic occurred during text normalization, using original text");
                text.to_string()
            })
        } else {
            text.to_string()
        };

        // Check again after normalization (normalization might have made it empty)
        if text.trim().is_empty() {
            return String::new();
        }

        // Map language codes
        let _lang_code = match self.lang.as_str() {
            "a" => "en-us",
            "b" => "en-gb",
            _ => "en-us",
        };

        // Get phonemes from backend
        let mut ps = match self.backend.phonemize(&text, _lang_code) {
            Some(phonemes) => {
                // Only warn if phonemes are empty AND input text is not empty
                // Empty input should legitimately return empty phonemes
                if phonemes.is_empty() && !text.trim().is_empty() {
                    tracing::warn!("⚠ Backend returned empty phonemes for text: '{}'", text);
                    tracing::warn!("   This may indicate the model is not working correctly!");
                }
                phonemes
            }
            None => {
                // Only error if input text is not empty
                if !text.trim().is_empty() {
                    tracing::error!("❌ CRITICAL: Backend returned None for text: '{}'", text);
                    tracing::error!(
                        "   This means the G2P model failed to load or process the text!"
                    );
                    tracing::error!("   Audio output will be incorrect! Check model loading logs.");
                }
                // Return empty string as fallback
                String::new()
            }
        };

        // Apply kokoro-specific replacements
        ps = ps
            .replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ")
            .replace("kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ");

        // Apply character replacements
        ps = ps
            .replace("ʲ", "j")
            .replace("r", "ɹ")
            .replace("x", "k")
            .replace("ɬ", "l");

        // Apply regex patterns
        // Insert space before "hˈʌndɹɪd" when it follows certain characters
        ps = PHONEME_PATTERNS.replace_all(&ps, "$1 $2").to_string();
        // Fix " z" before punctuation - replace " z" + punctuation with "z" + punctuation
        ps = Z_PATTERN.replace_all(&ps, "z$1").to_string();

        if self.lang == "a" {
            // Fix "ninti" -> "nindi" for US English
            ps = NINETY_PATTERN.replace_all(&ps, "$1di$2").to_string();
        }

        // Filter characters present in vocabulary
        ps = ps.chars().filter(|&c| VOCAB.contains_key(&c)).collect();

        ps.trim().to_string()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_misaki_abbreviations() {
        let phonemizer = Phonemizer::new("a", "ignored_path").unwrap();
        let cases = vec![
            "I'll", "I've", "it's", "he's", "she's", "we're", "they're",
            "isn't", "aren't", "wasn't", "weren't",
            "don't", "doesn't", "didn't",
            "can't", "couldn't", "shouldn't", "wouldn't", "won't",
            "hasn't", "haven't", "hadn't",
            "let's", "that's", "what's", "who's", "here's", "there's", "where's",
            "how's",
        ];
        
        for text in cases {
            let p = phonemizer.phonemize(text, true); // true = normalize
            println!("'{}' -> '{}'", text, p);
            assert!(!p.is_empty(), "Phonemes empty for '{}'", text);
            // Verify that we didn't lose the contraction phonemes due to normalization stripping '
            // e.g. "don't" should not sound like "dont" (unknown)
            // misaki returns IPA.
        }
    }
}
