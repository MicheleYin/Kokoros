use crate::tts::normalize;
use crate::tts::vocab::VOCAB;
#[cfg(feature = "espeak")]
use espeak_rs::text_to_phonemes;
use lazy_static::lazy_static;
use regex::Regex;
use std::path::PathBuf;
use voirs_g2p::backends::rule_based::RuleBasedG2p;
use voirs_g2p::{G2p, LanguageCode};
use futures::executor::block_on;

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

// Global mutex to serialize espeak-rs calls to prevent phoneme randomization
#[cfg(feature = "espeak")]
lazy_static::lazy_static! {
    static ref ESPEAK_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
}

/// Backend type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Espeak,
    /// Rule-based G2P backend using voirs-g2p
    VoirsG2p,
}

/// Espeak backend implementation
#[cfg(feature = "espeak")]
struct EspeakBackend {
    preserve_punctuation: bool,
    with_stress: bool,
}

#[cfg(feature = "espeak")]
impl EspeakBackend {
    fn new(preserve_punctuation: bool, with_stress: bool) -> Self {
        EspeakBackend {
            preserve_punctuation,
            with_stress,
        }
    }

    fn phonemize(&self, text: &str, language: &str) -> Option<String> {
        let _guard = ESPEAK_MUTEX.lock().unwrap();
        text_to_phonemes(text, language, None, self.preserve_punctuation, self.with_stress)
            .ok()
            .map(|phonemes| phonemes.join(""))
    }
    
    /// Check if espeak is available
    fn is_available() -> bool {
        // If espeak-rs is compiled in, it should be available
        // We can't really test at compile time, so we assume it's available if the feature is enabled
        true
    }
}

/// Check if espeak is available at runtime
#[cfg(not(feature = "espeak"))]
fn espeak_available() -> bool {
    false
}

#[cfg(feature = "espeak")]
fn espeak_available() -> bool {
    EspeakBackend::is_available()
}

/// RuleBasedG2p backend implementation using voirs-g2p
struct RuleBasedG2pBackend {
    model: Option<RuleBasedG2p>,
    language_code: Option<LanguageCode>,
}

impl RuleBasedG2pBackend {
    fn new(language_code: Option<LanguageCode>) -> Self {
        let model = language_code.map(|lang| RuleBasedG2p::new(lang));
        
        RuleBasedG2pBackend { 
            model,
            language_code,
        }
    }

   
    /// Convert language string to LanguageCode
    fn language_str_to_code(lang: &str) -> Option<LanguageCode> {
        match lang.to_lowercase().as_str() {
            "a" | "en" | "en-us" | "en_us" => Some(LanguageCode::EnUs),
            "b" | "en-gb" | "en_gb" => Some(LanguageCode::EnGb),
            _ => None,
        }
    }

    fn phonemize(&self, text: &str, language: &str) -> Option<String> {
        // Get language code - use stored one or try to parse from language string
        let lang_code = self.language_code
            .or_else(|| Self::language_str_to_code(language))?;
        
        // Get model - if we don't have one, create a temporary one
        // Since RuleBasedG2p is lightweight, we can create it on the fly if needed
        let model = if let Some(ref m) = self.model {
            m
        } else {
            // Create a temporary model for this call
            // This is acceptable since RuleBasedG2p is rule-based and doesn't load large models
            return None; // For now, return None if model not initialized
        };
        
        // Split text into words and phonemize each word
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut phonemes = Vec::new();
        
        for word in words {
            // Remove punctuation for phonemization
            let clean_word: String = word.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect();
            
            if clean_word.is_empty() {
                continue;
            }
            
            // Call async to_phonemes and block on it
            match block_on(model.to_phonemes(&clean_word, Some(lang_code))) {
                Ok(phoneme_vec) => {
                    // Convert Vec<Phoneme> to string by extracting the symbol field
                    let phoneme_string: String = phoneme_vec.iter()
                        .map(|p| {
                            // Extract the symbol from the Phoneme struct
                            // Phoneme has a public `symbol` field containing the phoneme character
                            p.symbol.as_str()
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    
                    if !phoneme_string.is_empty() {
                        phonemes.push(phoneme_string);
                    }
                }
                Err(e) => {
                    tracing::debug!("Failed to phonemize word '{}': {:?}", clean_word, e);
                    // Continue with other words even if one fails
                }
            }
        }
        
        if phonemes.is_empty() {
            None
        } else {
            // Join word phonemes with spaces
            Some(phonemes.join(" "))
        }
    }
}

/// Enum to hold different backend implementations
enum Backend {
    #[cfg(feature = "espeak")]
    Espeak(EspeakBackend),
    RuleBasedG2p(RuleBasedG2pBackend),
}

impl Backend {
    fn phonemize(&self, text: &str, language: &str) -> Option<String> {
        match self {
            #[cfg(feature = "espeak")]
            Backend::Espeak(backend) => backend.phonemize(text, language),
            Backend::RuleBasedG2p(backend) => backend.phonemize(text, language),
        }
    }

    fn set_model_path(&mut self, _path: PathBuf) {
        // RuleBasedG2p doesn't use model paths, so this is a no-op
        // But we keep the method for API compatibility
    }
}

pub struct Phonemizer {
    lang: String,
    backend: Backend,
    backend_type: BackendType,
}

impl Phonemizer {
    /// Create a new Phonemizer with Espeak backend (default if available, otherwise falls back to VoirsG2p)
    pub fn new(lang: &str) -> Self {
        // Try espeak first if available, otherwise fallback to VoirsG2p
        if cfg!(feature = "espeak") && espeak_available() {
            Self::new_with_backend(lang, BackendType::Espeak, None)
        } else {
            Self::new_with_backend(lang, BackendType::VoirsG2p, None)
        }
    }

    /// Create a new Phonemizer with specified backend
    pub fn new_with_backend(
        lang: &str,
        backend_type: BackendType,
        _model_path: Option<PathBuf>,
    ) -> Self {
        let backend = match backend_type {
            #[cfg(feature = "espeak")]
            BackendType::Espeak => {
                if espeak_available() {
                    Backend::Espeak(EspeakBackend::new(true, true))
                } else {
                    // Fallback to VoirsG2p if espeak requested but not available
                    let lang_code = RuleBasedG2pBackend::language_str_to_code(lang);
                    Backend::RuleBasedG2p(RuleBasedG2pBackend::new(lang_code))
                }
            }
            #[cfg(not(feature = "espeak"))]
            BackendType::Espeak => {
                // Espeak not compiled in, fallback to VoirsG2p
                let lang_code = RuleBasedG2pBackend::language_str_to_code(lang);
                Backend::RuleBasedG2p(RuleBasedG2pBackend::new(lang_code))
            }
            BackendType::VoirsG2p => {
                // Convert language string to LanguageCode
                let lang_code = RuleBasedG2pBackend::language_str_to_code(lang);
                Backend::RuleBasedG2p(RuleBasedG2pBackend::new(lang_code))
            }
        };

        Phonemizer {
            lang: lang.to_string(),
            backend,
            backend_type,
        }
    }

    /// Create a new Phonemizer with RuleBasedG2p backend
    /// Note: RuleBasedG2p doesn't require model files, so model_dir is ignored
    pub fn new_with_g2p(lang: &str, _model_dir: Option<&std::path::Path>) -> Self {
        Self::new_with_backend(lang, BackendType::VoirsG2p, None)
    }

    /// Get the current backend type
    pub fn backend_type(&self) -> BackendType {
        self.backend_type
    }

    /// Set model path for RuleBasedG2p backend
    /// Note: RuleBasedG2p doesn't use model paths, so this is a no-op
    /// Kept for API compatibility
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
        let espeak_lang = match self.lang.as_str() {
            "a" => "en-us",
            "b" => "en-gb",
            _ => "en-us",
        };

        // Get phonemes from backend
        let mut ps = self
            .backend
            .phonemize(&text, espeak_lang)
            .unwrap_or_else(|| {
                tracing::warn!("Backend returned None, using empty string");
                String::new()
            });

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
