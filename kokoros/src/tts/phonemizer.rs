use crate::tts::normalize;
use crate::tts::vocab::VOCAB;
use lazy_static::lazy_static;
use regex::Regex;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use piper_tts_rust::PhonemeGen;

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
    /// piper-tts-rust backend (uses mini-bart-g2p ONNX model)
    PiperTtsRust,
}

/// piper-tts-rust backend implementation (uses mini-bart-g2p ONNX model)
struct PiperTtsRustBackend {
    phoneme_gen: Option<Arc<Mutex<PhonemeGen>>>,
}

impl PiperTtsRustBackend {
    fn new(model_path: Option<PathBuf>) -> Self {
        let model_dir = if let Some(ref path) = model_path {
            // Canonicalize the provided path
            match path.canonicalize() {
                Ok(p) => p,
                Err(e) => {
                    tracing::error!("❌ Failed to canonicalize model path {:?}: {:?}", path, e);
                    PathBuf::from("")
                }
            }
        } else {
            // Try default paths (relative to workspace root)
            let default_paths = vec![
                PathBuf::from("../../mini-bart-g2p"),  // From kokoros/kokoros/
                PathBuf::from("../mini-bart-g2p"),     // Alternative
                PathBuf::from("mini-bart-g2p"),        // If in workspace root
            ];
            
            let mut found = None;
            for default_path in default_paths {
                // Try to canonicalize to resolve relative paths
                let resolved = if default_path.is_relative() {
                    // Try to resolve relative to current working directory
                    std::env::current_dir()
                        .ok()
                        .and_then(|cwd| cwd.join(&default_path).canonicalize().ok())
                        .or_else(|| default_path.canonicalize().ok())
                } else {
                    default_path.canonicalize().ok()
                };
                
                if let Some(resolved_path) = resolved {
                    if resolved_path.join("tokenizer.json").exists() 
                        && resolved_path.join("onnx/encoder_model.onnx").exists()
                        && resolved_path.join("onnx/decoder_model.onnx").exists() {
                        found = Some(resolved_path);
                        break;
                    }
                }
            }
            
            found.unwrap_or_else(|| {
                tracing::warn!("⚠ mini-bart-g2p model not found at any default path. Please specify model_path when creating the backend.");
                PathBuf::from("")
            })
        };
        
        let phoneme_gen = if !model_dir.as_os_str().is_empty() {
            let decoder_path = model_dir.join("onnx/decoder_model.onnx");
            let encoder_path = model_dir.join("onnx/encoder_model.onnx");
            let tokenizer_path = model_dir.join("tokenizer.json");
            let vocab_path = model_dir.join("vocab.json");
            
            // Find arpabet mapping - try multiple locations relative to model_dir
            let arpabet_mapping_path = {
                let candidates = vec![
                    model_dir.parent().map(|p| p.join("piper-tts-rust/arpabet-mapping.txt")),
                    model_dir.parent().and_then(|p| p.parent()).map(|p| p.join("piper-tts-rust/arpabet-mapping.txt")),
                    Some(PathBuf::from("../../piper-tts-rust/arpabet-mapping.txt")),
                    Some(PathBuf::from("../piper-tts-rust/arpabet-mapping.txt")),
                    Some(PathBuf::from("piper-tts-rust/arpabet-mapping.txt")),
                ];
                
                let mut found_mapping = None;
                for candidate in candidates.into_iter().flatten() {
                    let resolved = if candidate.is_relative() {
                        std::env::current_dir()
                            .ok()
                            .and_then(|cwd| cwd.join(&candidate).canonicalize().ok())
                            .or_else(|| candidate.canonicalize().ok())
                    } else {
                        candidate.canonicalize().ok()
                    };
                    
                    if let Some(path) = resolved {
                        if path.exists() {
                            found_mapping = Some(path);
                            break;
                        }
                    }
                }
                
                found_mapping.unwrap_or_else(|| {
                    tracing::warn!("⚠ arpabet-mapping.txt not found, using fallback path");
                    PathBuf::from("../../piper-tts-rust/arpabet-mapping.txt")
                })
            };
            
            // Verify all required files exist
            if !decoder_path.exists() {
                tracing::error!("❌ Decoder model not found at: {:?}", decoder_path);
                None
            } else if !encoder_path.exists() {
                tracing::error!("❌ Encoder model not found at: {:?}", encoder_path);
                None
            } else if !tokenizer_path.exists() {
                tracing::error!("❌ Tokenizer not found at: {:?}", tokenizer_path);
                None
            } else if !vocab_path.exists() {
                tracing::error!("❌ Vocab not found at: {:?}", vocab_path);
                None
            } else if !arpabet_mapping_path.exists() {
                tracing::error!("❌ ARPABET mapping not found at: {:?}", arpabet_mapping_path);
                None
            } else {
                tracing::info!("✓ Loading piper-tts-rust model from: {:?}", model_dir);
                tracing::debug!("  - Encoder: {:?}", encoder_path);
                tracing::debug!("  - Decoder: {:?}", decoder_path);
                tracing::debug!("  - Tokenizer: {:?}", tokenizer_path);
                tracing::debug!("  - Vocab: {:?}", vocab_path);
                tracing::debug!("  - ARPABET mapping: {:?}", arpabet_mapping_path);
                
                let mut phoneme_gen = PhonemeGen::new(
                    decoder_path.to_string_lossy().to_string(),
                    encoder_path.to_string_lossy().to_string(),
                    tokenizer_path.to_string_lossy().to_string(),
                    vocab_path.to_string_lossy().to_string(),
                    arpabet_mapping_path.to_string_lossy().to_string(),
                );
                
                match phoneme_gen.load() {
                    Ok(_) => {
                        tracing::info!("✓ piper-tts-rust PhonemeGen loaded successfully from {:?}", model_dir);
                        Some(Arc::new(Mutex::new(phoneme_gen)))
            }
            Err(e) => {
                        tracing::error!("❌ Failed to load piper-tts-rust PhonemeGen from {:?}: {:?}", model_dir, e);
                        tracing::error!("   Checked paths:");
                        tracing::error!("     - Encoder: {:?} (exists: {})", encoder_path, encoder_path.exists());
                        tracing::error!("     - Decoder: {:?} (exists: {})", decoder_path, decoder_path.exists());
                        tracing::error!("     - Tokenizer: {:?} (exists: {})", tokenizer_path, tokenizer_path.exists());
                        tracing::error!("     - Vocab: {:?} (exists: {})", vocab_path, vocab_path.exists());
                        tracing::error!("     - ARPABET mapping: {:?} (exists: {})", arpabet_mapping_path, arpabet_mapping_path.exists());
                None
            }
                }
            }
        } else {
            None
        };
        
        PiperTtsRustBackend {
            phoneme_gen,
        }
    }

    fn phonemize(&self, text: &str, _language: &str) -> Option<String> {
        let phoneme_gen_arc = match self.phoneme_gen.as_ref() {
            Some(pg) => pg,
            None => {
                tracing::error!("❌ piper-tts-rust PhonemeGen not initialized! Model may have failed to load.");
                return None;
            }
        };
        
        // Split text into words and phonemize each word
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut phonemes = Vec::new();
        
        for word in words {
            // Remove punctuation for phonemization
            let clean_word: String = word.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'' || *c == '-')
                .collect();
            
            if clean_word.is_empty() {
                continue;
            }
            
            // Convert word to IPA using piper-tts-rust
            let mut phoneme_gen_guard = phoneme_gen_arc.lock().unwrap();
            match phoneme_gen_guard.process_word(&clean_word) {
                Ok(ipa_phonemes) => {
                    if !ipa_phonemes.is_empty() {
                        tracing::debug!("piper-tts-rust: '{}' -> '{}'", clean_word, ipa_phonemes.join(""));
                        // Join IPA phonemes without spaces
                        let phoneme_str = ipa_phonemes.join("");
                        phonemes.push(phoneme_str);
                    } else {
                        tracing::warn!("piper-tts-rust returned empty phonemes for '{}' - this may indicate a model issue", clean_word);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to phonemize word '{}' with piper-tts-rust: {:?}", clean_word, e);
                    // Continue with other words even if one fails
                }
            }
            drop(phoneme_gen_guard); // Release lock early
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
    PiperTtsRust(PiperTtsRustBackend),
}

impl Backend {
    fn phonemize(&self, text: &str, language: &str) -> Option<String> {
        match self {
            Backend::PiperTtsRust(backend) => backend.phonemize(text, language),
        }
    }

    fn set_model_path(&mut self, path: PathBuf) {
        match self {
            Backend::PiperTtsRust(backend) => {
                // Reload model with new path
                *backend = PiperTtsRustBackend::new(Some(path));
            }
            _ => {
                // Other backends don't use model paths, so this is a no-op
            }
        }
    }
}

pub struct Phonemizer {
    lang: String,
    backend: Backend,
    backend_type: BackendType,
}

impl Phonemizer {
    /// Create a new Phonemizer with PiperTtsRust backend (mini-bart-g2p)
    pub fn new(lang: &str) -> Self {
        // Use PiperTtsRust (mini-bart-g2p)
        // Try to find model in common locations
        let model_path = Self::find_default_model_path();
        Self::new_with_backend(lang, BackendType::PiperTtsRust, model_path)
    }
    
    /// Find the default mini-bart-g2p model path in common locations
    fn find_default_model_path() -> Option<PathBuf> {
        // Try to resolve paths relative to current working directory
        let current_dir = std::env::current_dir().ok();
        
        let possible_paths: Vec<PathBuf> = {
            let mut paths = vec![
                // Tauri resource directory (production - relative to resource dir)
                PathBuf::from("resources/mini-bart-g2p"),
                // Development paths
                PathBuf::from("src-tauri/resources/mini-bart-g2p"),
                PathBuf::from("../../mini-bart-g2p"),
                PathBuf::from("../mini-bart-g2p"),
                PathBuf::from("mini-bart-g2p"),
            ];
            
            // Add absolute paths if we have a current directory
            if let Some(ref cwd) = current_dir {
                paths.push(cwd.join("src-tauri/resources/mini-bart-g2p"));
                paths.push(cwd.join("resources/mini-bart-g2p"));
                paths.push(cwd.join("mini-bart-g2p"));
                // Also try parent directories
                if let Some(parent) = cwd.parent() {
                    paths.push(parent.join("mini-bart-g2p"));
                }
            }
            paths
        };
        
        for path in possible_paths {
            // Try to canonicalize the path
            let resolved = if path.is_relative() {
                current_dir.as_ref()
                    .and_then(|cwd| cwd.join(&path).canonicalize().ok())
                    .or_else(|| path.canonicalize().ok())
            } else {
                path.canonicalize().ok()
            };
            
            let check_path = resolved.as_ref().unwrap_or(&path);
            
            if check_path.join("tokenizer.json").exists() 
                && check_path.join("onnx/encoder_model.onnx").exists()
                && check_path.join("onnx/decoder_model.onnx").exists() {
                return resolved.or(Some(path));
            }
        }
        None
    }

    /// Create a new Phonemizer with specified backend
    pub fn new_with_backend(
        lang: &str,
        backend_type: BackendType,
        model_path: Option<PathBuf>,
    ) -> Self {
        let backend = match backend_type {
            BackendType::PiperTtsRust => {
                Backend::PiperTtsRust(PiperTtsRustBackend::new(model_path))
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

        // Map language codes (for compatibility, though piper-tts-rust uses its own language handling)
        let _lang_code = match self.lang.as_str() {
            "a" => "en-us",
            "b" => "en-gb",
            _ => "en-us",
        };

        // Get phonemes from backend
        let mut ps = self
            .backend
            .phonemize(&text, _lang_code)
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
