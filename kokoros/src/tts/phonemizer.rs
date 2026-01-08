use crate::tts::normalize;
use crate::tts::vocab::VOCAB;
use lazy_static::lazy_static;
use piper_tts_rust::PhonemeGen;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

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
    /// ONNX phonemizer backend (uses mini-bart-g2p model)
    Onnx,
}

/// ONNX backend implementation (uses mini-bart-g2p model via piper-tts-rust)
struct OnnxBackend {
    phoneme_gen: Mutex<PhonemeGen>,
    arpabet_to_ipa: HashMap<String, String>,
}

impl OnnxBackend {
    fn new(model_dir: &std::path::Path) -> Result<Self, String> {
        let encoder_path = model_dir.join("onnx").join("encoder_model.onnx");
        let decoder_path = model_dir.join("onnx").join("decoder_model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let vocab_path = model_dir.join("vocab.json");
        let arpabet_mapping_path = model_dir.join("arpabet-mapping.txt");

        if !encoder_path.exists() {
            return Err(format!("Encoder model not found: {:?}", encoder_path));
        }
        if !decoder_path.exists() {
            return Err(format!("Decoder model not found: {:?}", decoder_path));
        }
        if !tokenizer_path.exists() {
            return Err(format!("Tokenizer not found: {:?}", tokenizer_path));
        }
        if !vocab_path.exists() {
            return Err(format!("Vocab not found: {:?}", vocab_path));
        }
        if !arpabet_mapping_path.exists() {
            return Err(format!(
                "ARPAbet mapping not found: {:?}",
                arpabet_mapping_path
            ));
        }

        tracing::info!("Initializing ONNX phonemizer from: {:?}", model_dir);

        // Initialize PhonemeGen from piper-tts-rust
        let mut phoneme_gen = PhonemeGen::new(
            decoder_path.to_string_lossy().to_string(),
            encoder_path.to_string_lossy().to_string(),
            tokenizer_path.to_string_lossy().to_string(),
            vocab_path.to_string_lossy().to_string(),
            arpabet_mapping_path.to_string_lossy().to_string(),
        );

        phoneme_gen
            .load()
            .map_err(|e| format!("Failed to load PhonemeGen: {}", e))?;

        // Load ARPAbet to IPA mapping
        let arpabet_to_ipa = Self::load_arpabet_mapping(&arpabet_mapping_path)?;

        tracing::info!("✓ ONNX phonemizer backend initialized successfully");
        tracing::info!("  Loaded {} ARPAbet to IPA mappings", arpabet_to_ipa.len());

        Ok(Self {
            phoneme_gen: Mutex::new(phoneme_gen),
            arpabet_to_ipa,
        })
    }

    /// Load ARPAbet to IPA mapping from file
    fn load_arpabet_mapping(path: &std::path::Path) -> Result<HashMap<String, String>, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read ARPAbet mapping file: {}", e))?;

        let mut mapping = HashMap::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let arpabet = parts[0].to_string();
                let ipa = parts[1..].join(" "); // Join in case IPA has spaces
                // Store both uppercase and lowercase versions for case-insensitive matching
                mapping.insert(arpabet.clone(), ipa.clone());
                mapping.insert(arpabet.to_lowercase(), ipa.clone());
                mapping.insert(arpabet.to_uppercase(), ipa);
            }
        }

        Ok(mapping)
    }

    /// Convert ARPAbet notation to IPA
    fn convert_arpabet_to_ipa(&self, arpabet: &str) -> String {
        // The piper-tts-rust model outputs ARPAbet with underscores: "^_H_H_E_H_1_L_O_W_0_,_ _W_E_R_1_L_D_!_$"
        // Format: ^ (start) _ (separator) phoneme_parts _ (separator) ... spaces are preserved as " _ " (space between underscores)
        // Remove start/end markers and process preserving spaces
        tracing::debug!("Converting ARPAbet to IPA: '{}'", arpabet);

        let cleaned = arpabet.trim_start_matches('^').trim_end_matches('$');

        // Split by underscores, but preserve empty strings which indicate spaces
        // Pattern: "_X_Y_" -> ["", "X", "Y", ""] where empty strings represent word boundaries
        let parts: Vec<&str> = cleaned.split('_').collect();

        if parts.is_empty() {
            return String::new();
        }

        // Reconstruct ARPAbet phonemes from parts and convert to IPA
        // Spaces appear as a single space character " " in the parts array
        let mut result = String::new();
        let mut i = 0;

        while i < parts.len() {
            // Check if this is a space character
            if parts[i] == " " {
                result.push(' ');
                i += 1;
                continue;
            }

            // Skip empty parts (they're just separators)
            if parts[i].is_empty() {
                i += 1;
                continue;
            }

            // Try to match longest possible phoneme (up to 4 parts)
            let mut matched = false;

            // Try matching from longest to shortest (4 parts down to 1 part)
            for len in (1..=parts.len().saturating_sub(i).min(4)).rev() {
                // Collect parts for this potential phoneme (skip empty parts and spaces)
                let mut phoneme_parts = Vec::new();
                let mut j = i;
                let mut collected = 0;

                while j < parts.len() && collected < len {
                    if !parts[j].is_empty() && parts[j] != " " {
                        phoneme_parts.push(parts[j]);
                        collected += 1;
                    }
                    j += 1;
                    // Stop if we hit a space (word boundary)
                    if j < parts.len() && parts[j] == " " {
                        break;
                    }
                }

                if phoneme_parts.len() == len {
                    let phoneme = phoneme_parts.join("");

                    let variants = vec![
                        phoneme.clone(),
                        phoneme.to_uppercase(),
                        phoneme.to_lowercase(),
                    ];

                    for variant in &variants {
                        if let Some(ipa) = self.arpabet_to_ipa.get(variant) {
                            result.push_str(ipa);
                            // Advance past all parts we used
                            i = j;
                            matched = true;
                            tracing::debug!("Matched phoneme '{}' -> '{}'", phoneme, ipa);
                            break;
                        }
                    }

                    if matched {
                        break;
                    }
                }
            }

            if !matched {
                // Check if it's punctuation
                let part = parts[i];
                if part.len() == 1 {
                    if let Some(c) = part.chars().next() {
                        if ".,!?:;".contains(c) {
                            result.push(c);
                            i += 1;
                            continue;
                        }
                    }
                }

                tracing::debug!("Skipping unmatched part: '{}'", parts[i]);
                i += 1;
            }
        }

        tracing::debug!("Converted '{}' -> '{}'", arpabet, result);
        result
    }

    #[allow(dead_code)]
    fn _old_convert_arpabet_to_ipa_unused(&self, _arpabet: &str) -> String {
        // Old implementation - kept for reference
        let mut result = String::new();
        let mut remaining = _arpabet;

        while !remaining.is_empty() {
            // Skip whitespace
            if remaining.starts_with(char::is_whitespace) {
                result.push(' ');
                remaining = remaining.trim_start();
                continue;
            }

            // Try to match the longest possible ARPAbet phoneme (1-4 characters)
            let mut matched = false;
            let mut matched_ipa = String::new();
            let mut matched_len = 0;

            // Try matching from longest to shortest (4 chars down to 1 char)
            // This ensures we match "EH1" before "E" or "EH"
            for len in (1..=remaining.len().min(4)).rev() {
                if let Some(phoneme) = remaining.get(0..len) {
                    // Strategy: Try multiple variations
                    // 1. Exact match (case-sensitive)
                    // 2. Uppercase version
                    // 3. Lowercase version
                    // 4. If no stress marker (0/1/2), try adding stress markers

                    let mut variants_to_try = vec![
                        phoneme.to_string(),
                        phoneme.to_uppercase(),
                        phoneme.to_lowercase(),
                    ];

                    // Also try adding stress markers if phoneme doesn't end with a digit
                    if !phoneme.chars().last().map_or(false, |c| c.is_ascii_digit()) {
                        let upper = phoneme.to_uppercase();
                        let lower = phoneme.to_lowercase();
                        // Try adding stress markers 0, 1, 2 (prefer 1 for primary stress)
                        for stress in &["1", "0", "2"] {
                            variants_to_try.push(format!("{}{}", upper, stress));
                            variants_to_try.push(format!("{}{}", lower, stress));
                        }
                    }

                    for variant in &variants_to_try {
                        if let Some(ipa) = self.arpabet_to_ipa.get(variant) {
                            matched_ipa = ipa.clone();
                            matched_len = len;
                            matched = true;
                            tracing::debug!(
                                "Matched '{}' -> '{}' (variant: '{}')",
                                phoneme,
                                ipa,
                                variant
                            );
                            break;
                        }
                    }

                    if matched {
                        break;
                    }
                }
            }

            if matched {
                result.push_str(&matched_ipa);
                remaining = &remaining[matched_len..];
            } else {
                // No match found - check if it's punctuation or special character
                let next_char = remaining.chars().next();
                if let Some(c) = next_char {
                    if ".,!?:;-$".contains(c) {
                        // Keep punctuation and special characters as-is
                        result.push(c);
                        remaining = &remaining[c.len_utf8()..];
                    } else {
                        // Unknown character - log and keep as-is
                        tracing::warn!(
                            "Unknown ARPAbet character '{}' in '{}', keeping as-is",
                            c,
                            remaining
                        );
                        result.push(c);
                        remaining = &remaining[c.len_utf8()..];
                    }
                } else {
                    break;
                }
            }
        }

        tracing::debug!("Converted '{}' -> '{}'", _arpabet, result);
        result
    }

    fn phonemize(&self, text: &str, _language: &str) -> Option<String> {
        // Handle empty string case - return empty result without processing
        if text.trim().is_empty() {
            return Some(String::new());
        }

        // Process text through piper-tts-rust's phonemizer (returns ARPAbet)
        // Normalize text before processing to handle edge cases
        let normalized_text = text.to_lowercase();

        // Lock the mutex and process text
        // If a panic occurs, the mutex will be poisoned, but we handle that gracefully
        let arpabet_result = {
            let mut pg = match self.phoneme_gen.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    tracing::error!(
                        "Failed to lock phoneme_gen mutex (poisoned) - mutex was poisoned by a previous panic"
                    );
                    return None;
                }
            };

            // Wrap process_text in catch_unwind to prevent panics from poisoning the mutex
            // In production builds with aggressive optimizations, ONNX Runtime calls can cause
            // stack overflow panics. By catching the panic here, we prevent mutex poisoning
            // and allow subsequent calls to continue working.
            let process_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pg.process_text(&normalized_text)
            }));

            match process_result {
                Ok(Ok(arpabet_string)) => {
                    tracing::debug!("ARPAbet output for '{}': '{}'", text, arpabet_string);
                    arpabet_string
                }
                Ok(Err(e)) => {
                    tracing::warn!("Failed to phonemize text '{}': {:?}", text, e);
                    return None;
                }
                Err(panic_payload) => {
                    // Extract panic message if possible
                    let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        format!("{}", s)
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };

                    tracing::error!(
                        "Panic occurred during phonemization for text '{}': {}. \
                        This is often caused by 'No tokens generated' errors in piper-tts-rust \
                        (common with short/abbreviated text like 'O.', 'C.', etc.) or stack overflow \
                        in production builds with aggressive optimizations. \
                        Mutex was NOT poisoned due to catch_unwind protection - processing can continue.",
                        text,
                        panic_msg
                    );
                    return None;
                }
            }
        };

        // Convert ARPAbet to IPA
        let ipa_result = self.convert_arpabet_to_ipa(&arpabet_result);
        tracing::debug!(
            "IPA conversion for '{}': ARPAbet '{}' -> IPA '{}'",
            text,
            arpabet_result,
            ipa_result
        );
        Some(ipa_result)
    }

    fn is_initialized(&self) -> bool {
        true
    }
}

/// Enum to hold different backend implementations
enum Backend {
    Onnx(OnnxBackend),
}

impl Backend {
    fn phonemize(&self, text: &str, language: &str) -> Option<String> {
        match self {
            Backend::Onnx(backend) => backend.phonemize(text, language),
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
    /// Create a new Phonemizer with ONNX backend (uses mini-bart-g2p model)
    /// Uses ONNX as default
    /// Tries to find model in common locations if model_path is None
    /// Returns an error if the model cannot be found or initialized
    pub fn new(lang: &str) -> Result<Self, String> {
        // Try to find model in common locations
        let model_path = Self::find_model_path();
        Self::new_with_backend_result(lang, BackendType::Onnx, model_path)
    }

    /// Create a new Phonemizer, panicking on error (for backward compatibility)
    pub fn new_or_panic(lang: &str) -> Self {
        Self::new(lang).unwrap_or_else(|e| {
            tracing::error!("Failed to create Phonemizer: {}", e);
            panic!("Failed to create Phonemizer: {}", e);
        })
    }

    /// Find model path in common locations
    fn find_model_path() -> Option<PathBuf> {
        tracing::info!("Searching for mini-bart-g2p model...");

        // Check TAURI_RESOURCE_DIR first
        if let Ok(resource_dir) = std::env::var("TAURI_RESOURCE_DIR") {
            tracing::info!("  Checking TAURI_RESOURCE_DIR: {}", resource_dir);
            let path = PathBuf::from(&resource_dir).join("mini-bart-g2p");
            tracing::info!("    Checking path: {:?}", path);
            if path.exists() {
                tracing::info!("    Path exists: true");
                let encoder_path = path.join("onnx").join("encoder_model.onnx");
                tracing::info!("    Checking encoder: {:?}", encoder_path);
                if encoder_path.exists() {
                    tracing::info!("    ✓ Found model at: {:?}", path);
                    return Some(path);
                } else {
                    tracing::warn!("    ✗ Encoder model not found at: {:?}", encoder_path);
                }
            } else {
                tracing::warn!("    ✗ Path does not exist: {:?}", path);
            }
        } else {
            tracing::warn!("  TAURI_RESOURCE_DIR not set");
        }

        // Check current directory and common dev locations
        let current_dir = std::env::current_dir().ok();
        if let Some(ref cwd) = current_dir {
            tracing::info!("  Current working directory: {:?}", cwd);
        }

        let possible_paths = vec![
            PathBuf::from("src-tauri/resources/mini-bart-g2p"),
            PathBuf::from("resources/mini-bart-g2p"),
            PathBuf::from("../src-tauri/resources/mini-bart-g2p"),
            PathBuf::from("../../src-tauri/resources/mini-bart-g2p"),
        ];

        for path in &possible_paths {
            tracing::info!("  Checking path: {:?}", path);
            if path.exists() {
                tracing::info!("    Path exists: true");
                let encoder_path = path.join("onnx").join("encoder_model.onnx");
                if encoder_path.exists() {
                    tracing::info!("    ✓ Found model at: {:?}", path);
                    return Some(path.clone());
                } else {
                    tracing::warn!("    ✗ Encoder model not found at: {:?}", encoder_path);
                }
            } else {
                tracing::debug!("    Path does not exist: {:?}", path);
            }
        }

        tracing::error!("  ✗ Model not found in any location!");
        None
    }

    /// Create a new Phonemizer with specified backend (returns Result)
    pub fn new_with_backend_result(
        lang: &str,
        backend_type: BackendType,
        model_path: Option<PathBuf>,
    ) -> Result<Self, String> {
        let backend = match backend_type {
            BackendType::Onnx => {
                // Try to load ONNX backend with model path
                if let Some(ref path) = model_path {
                    match OnnxBackend::new(path) {
                        Ok(onnx_backend) => Backend::Onnx(onnx_backend),
                        Err(e) => {
                            let error_msg = format!("Failed to initialize ONNX backend: {}", e);
                            tracing::error!("{}", error_msg);
                            return Err(error_msg);
                        }
                    }
                } else {
                    let tauri_resource_dir = std::env::var("TAURI_RESOURCE_DIR").ok();
                    let current_dir = std::env::current_dir().ok();
                    let error_msg = format!(
                        "ONNX backend requires a model path. Set TAURI_RESOURCE_DIR or provide model_path.\n\
                        Current TAURI_RESOURCE_DIR: {:?}\n\
                        Current working directory: {:?}\n\
                        Expected model location: {:?} or {:?}",
                        tauri_resource_dir,
                        current_dir,
                        tauri_resource_dir
                            .as_ref()
                            .map(|d| PathBuf::from(d).join("mini-bart-g2p")),
                        current_dir
                            .as_ref()
                            .map(|d| d.join("src-tauri/resources/mini-bart-g2p"))
                    );
                    tracing::error!("{}", error_msg);
                    return Err(error_msg);
                }
            }
        };

        Ok(Phonemizer {
            lang: lang.to_string(),
            backend,
            backend_type,
            model_path,
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
