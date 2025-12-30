use crate::tts::normalize;
use crate::tts::vocab::VOCAB;
use lazy_static::lazy_static;
use regex::Regex;
use std::path::PathBuf;
use espeak_rs::text_to_phonemes;

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
    /// espeak-ng backend (uses piper-rs espeak-rs crate)
    Espeak,
}

/// Espeak backend implementation (uses piper-rs espeak-rs crate)
struct EspeakBackend;

impl EspeakBackend {
    fn new() -> Self {
        EspeakBackend
    }
    
    fn phonemize(&self, text: &str, language: &str) -> Option<String> {
        // Map language codes
        let espeak_lang = match language {
            "en" | "a" => "en",
            "en-us" | "en-US" => "en-us",
            "en-gb" | "en-GB" | "b" => "en-gb",
            _ => language,
        };
        
        // Use espeak-rs to get IPA phonemes
        match text_to_phonemes(text, espeak_lang, None, true, false) {
            Ok(phonemes) => {
                // Join phonemes into a string
                Some(phonemes.join(""))
            }
            Err(e) => {
                tracing::error!("Failed to phonemize with espeak-rs: {:?}", e);
                None
            }
        }
    }
    
    fn is_initialized(&self) -> bool {
        true // espeak-rs is always initialized via lazy static
    }
}



/// Enum to hold different backend implementations
enum Backend {
    Espeak(EspeakBackend),
}

impl Backend {
    fn phonemize(&self, text: &str, language: &str) -> Option<String> {
        match self {
            Backend::Espeak(backend) => backend.phonemize(text, language),
        }
    }

    fn set_model_path(&mut self, _path: PathBuf) {
        // Espeak backend doesn't use model paths
        tracing::warn!("set_model_path called on Espeak backend (no-op)");
    }
}

pub struct Phonemizer {
    lang: String,
    backend: Backend,
    backend_type: BackendType,
}

impl Phonemizer {
    /// Create a new Phonemizer with Espeak backend (uses piper-rs espeak-rs crate)
    /// Uses espeak-rs as default
    pub fn new(lang: &str) -> Self {
        // Use Espeak backend as default
        Self::new_with_backend(lang, BackendType::Espeak, None)
    }


    /// Create a new Phonemizer with specified backend
    pub fn new_with_backend(
        lang: &str,
        backend_type: BackendType,
        _model_path: Option<PathBuf>,
    ) -> Self {
        let backend = match backend_type {
            BackendType::Espeak => {
                Backend::Espeak(EspeakBackend::new())
            }
        };

        Phonemizer {
            lang: lang.to_string(),
            backend,
            backend_type,
        }
    }

    /// Get the current backend type
    pub fn backend_type(&self) -> BackendType {
        self.backend_type
    }

    /// Set model path (no-op, kept for API compatibility)
    pub fn set_model_path(&mut self, path: PathBuf) {
        self.backend.set_model_path(path);
    }

    pub fn phonemize(&self, text: &str, normalize: bool) -> String {
        let text = if normalize {
            normalize::normalize_text(text)
        } else {
            text.to_string()
        };

        // Map language codes
        let _lang_code = match self.lang.as_str() {
            "a" => "en-us",
            "b" => "en-gb",
            _ => "en-us",
        };

        // Get phonemes from backend
        let mut ps = match self.backend.phonemize(&text, _lang_code) {
            Some(phonemes) => {
                if phonemes.is_empty() {
                    tracing::warn!("⚠ Backend returned empty phonemes for text: '{}'", text);
                    tracing::warn!("   This may indicate the model is not working correctly!");
                }
                phonemes
            }
            None => {
                tracing::error!("❌ CRITICAL: Backend returned None for text: '{}'", text);
                tracing::error!("   This means the G2P model failed to load or process the text!");
                tracing::error!("   Audio output will be incorrect! Check model loading logs.");
                // Return empty string as fallback, but log the error
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
