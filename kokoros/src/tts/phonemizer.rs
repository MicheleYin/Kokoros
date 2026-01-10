use crate::tts::normalize;
use crate::tts::vocab::VOCAB;
use lazy_static::lazy_static;
use piper_tts_rust::PhonemeGen;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

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

// Global phonemizer singleton - only used by new_auto() for backward compatibility
// The main new() method doesn't use singleton (like TTS OrtKoko)
static GLOBAL_PHONEMIZER: OnceLock<Mutex<Option<Arc<Phonemizer>>>> = OnceLock::new();

impl OnnxBackend {
    fn new(model_dir: &std::path::Path) -> Result<Self, String> {
        let encoder_path = model_dir.join("onnx").join("encoder_model.onnx");
        let decoder_path = model_dir.join("onnx").join("decoder_model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let vocab_path = model_dir.join("vocab.json");
        let arpabet_mapping_path = model_dir.join("arpabet-mapping.txt");

        // Validate file existence and accessibility BEFORE attempting to load
        // In production, bundled files might have different access patterns
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

        // Verify files are readable (important for bundled files in production)
        // ONNX Runtime might fail silently or abort() if files aren't accessible
        let test_read = std::fs::metadata(&encoder_path).map_err(|e| {
            format!(
                "Cannot access encoder model file: {} - {}",
                encoder_path.display(),
                e
            )
        })?;
        if test_read.len() == 0 {
            return Err(format!("Encoder model file is empty: {:?}", encoder_path));
        }

        tracing::info!("Initializing ONNX phonemizer from: {:?}", model_dir);
        tracing::debug!(
            "Model file sizes - encoder: {} bytes, decoder: {} bytes",
            test_read.len(),
            std::fs::metadata(&decoder_path)
                .map(|m| m.len())
                .unwrap_or(0)
        );

        // Simple initialization (like TTS OrtKoko) - no locks needed
        // ONNX Runtime handles concurrent initialization internally (like ort crate does)
        // Each instance uses its own file paths, so no conflicts
        let _test_read_encoder = std::fs::read(&encoder_path).map_err(|e| {
            format!(
                "Cannot read encoder model file: {} - {}",
                encoder_path.display(),
                e
            )
        })?;
        let _test_read_decoder = std::fs::read(&decoder_path).map_err(|e| {
            format!(
                "Cannot read decoder model file: {} - {}",
                decoder_path.display(),
                e
            )
        })?;
        let _test_read_tokenizer = std::fs::read(&tokenizer_path).map_err(|e| {
            format!(
                "Cannot read tokenizer file: {} - {}",
                tokenizer_path.display(),
                e
            )
        })?;
        let _test_read_vocab = std::fs::read(&vocab_path)
            .map_err(|e| format!("Cannot read vocab file: {} - {}", vocab_path.display(), e))?;

        tracing::debug!("Verified all model files are readable, initializing PhonemeGen");

        // Initialize PhonemeGen from piper-tts-rust
        // Simple initialization (like TTS OrtKoko) - no locks, no catch_unwind
        // ONNX Runtime handles errors internally, and we validate files exist before loading
        let mut phoneme_gen = PhonemeGen::new(
            decoder_path.to_string_lossy().to_string(),
            encoder_path.to_string_lossy().to_string(),
            tokenizer_path.to_string_lossy().to_string(),
            vocab_path.to_string_lossy().to_string(),
            arpabet_mapping_path.to_string_lossy().to_string(),
        );

        // Load the model (load() modifies phoneme_gen in place, returns Result<(), Error>)
        phoneme_gen.load().map_err(|e| {
            format!(
                "Failed to load PhonemeGen model: {}. \
                This may indicate:\n\
                - Corrupted ONNX model files (encoder_model.onnx, decoder_model.onnx)\n\
                - Corrupted tokenizer or vocab files\n\
                - ONNX Runtime initialization failure\n\
                - Insufficient memory\n\
                - Incompatible ONNX Runtime version\n\
                \n\
                Model directory: {:?}",
                e, model_dir
            )
        })?;

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

        // Aggressive input validation to prevent ONNX Runtime abort() calls
        // ONNX Runtime can call abort() directly from C++ if it receives invalid input
        // We need to filter out anything that might cause issues before it reaches ONNX Runtime
        let normalized_text = text.to_lowercase();

        // Filter out problematic characters that might cause ONNX Runtime to abort
        // Keep only alphanumeric, whitespace, and basic punctuation
        let filtered_text: String = normalized_text
            .chars()
            .filter(|c| {
                c.is_alphanumeric()
                    || c.is_whitespace()
                    || matches!(c, '.' | ',' | '!' | '?' | ':' | ';' | '-' | '\'' | '"')
            })
            .collect();

        // If filtering removed everything, return empty
        if filtered_text.trim().is_empty() {
            tracing::warn!(
                "Text '{}' was filtered to empty, skipping phonemization",
                text
            );
            return Some(String::new());
        }

        // Limit text length to prevent potential buffer overflows or memory issues
        // piper-tts-rust/ONNX Runtime might have internal limits
        const MAX_PHONEMIZE_LENGTH: usize = 1000;
        let text_to_process = if filtered_text.len() > MAX_PHONEMIZE_LENGTH {
            tracing::warn!(
                "Text length {} exceeds max {} chars, truncating for phonemization",
                filtered_text.len(),
                MAX_PHONEMIZE_LENGTH
            );
            &filtered_text[..MAX_PHONEMIZE_LENGTH]
        } else {
            &filtered_text
        };

        // Lock the mutex and process text
        // Release the lock as quickly as possible to avoid holding it during ONNX Runtime calls
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
            // NOTE: This won't catch C++ abort() calls from ONNX Runtime, but will catch Rust panics
            // If ONNX Runtime calls abort(), the entire process will terminate regardless
            let process_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pg.process_text(text_to_process)
            }));

            // Release the mutex lock immediately after processing
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
    ///
    /// This method follows the same simple pattern as `OrtKoko::new()`:
    /// - Takes a direct path parameter (no path searching)
    /// - Simple initialization (no singleton complexity)
    /// - Returns Result<Self, String> (caller can wrap in Arc if needed)
    ///
    /// # Arguments
    /// * `lang` - Language code (e.g., "en")
    /// * `model_path` - Path to mini-bart-g2p model directory (e.g., "/path/to/mini-bart-g2p")
    ///
    /// # Example
    /// ```rust
    /// // Simple, direct initialization (like OrtKoko::new)
    /// let phonemizer = Phonemizer::new("en", "/path/to/mini-bart-g2p")?;
    /// let phonemizer = Arc::new(phonemizer); // Wrap in Arc if needed for sharing
    /// ```
    pub fn new(lang: &str, model_path: &str) -> Result<Self, String> {
        let model_path_buf = PathBuf::from(model_path);
        
        // Simple validation (like TTS model - just check exists)
        if !model_path_buf.exists() {
            return Err(format!(
                "G2P model directory not found: {}. \
                Make sure mini-bart-g2p model files are in the specified directory.",
                model_path
            ));
        }

        tracing::info!("Initializing G2P phonemizer from: {:?}", model_path_buf);
        
        // Validate required files exist (similar to TTS model validation)
        Self::validate_model_path(&model_path_buf)?;

        // Create phonemizer (simple, like OrtKoko::new)
        let phonemizer = Self::new_with_backend_result(lang, BackendType::Onnx, Some(model_path_buf))?;
        
        tracing::info!("✓ G2P phonemizer initialized successfully");
        Ok(phonemizer)
    }

    /// Create a new Phonemizer with automatic path detection (for backward compatibility)
    ///
    /// This method searches for the model in common locations and uses a singleton
    /// pattern for sharing. For production use, prefer `new()` with an explicit path.
    ///
    /// # Arguments
    /// * `lang` - Language code (e.g., "en")
    ///
    /// # Example
    /// ```rust
    /// // Auto-detect path (works in development, less reliable in production)
    /// let phonemizer = Phonemizer::new_auto("en")?;
    /// ```
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
        tracing::info!("Initializing global phonemizer singleton (auto-detecting path)");
        
        // Search for model path
        let found_path = Self::find_model_path();
        if found_path.is_none() {
            return Err(format!(
                "G2P model not found. Searched in:\n\
                - TAURI_RESOURCE_DIR/mini-bart-g2p\n\
                - src-tauri/resources/mini-bart-g2p\n\
                - resources/mini-bart-g2p\n\
                - ../src-tauri/resources/mini-bart-g2p\n\
                - ../../src-tauri/resources/mini-bart-g2p\n\
                \n\
                For production, use Phonemizer::new() with an explicit path.\n\
                Current TAURI_RESOURCE_DIR: {:?}\n\
                Current working directory: {:?}",
                std::env::var("TAURI_RESOURCE_DIR").ok(),
                std::env::current_dir().ok()
            ));
        }

        let phonemizer = Self::new_with_backend_result(lang, BackendType::Onnx, found_path)?;
        let phonemizer_arc = Arc::new(phonemizer);

        // Store in global singleton
        *phonemizer_opt = Some(Arc::clone(&phonemizer_arc));
        tracing::info!("✓ Global phonemizer singleton initialized and cached");

        Ok(phonemizer_arc)
    }

    /// Create a new Phonemizer, panicking on error (for backward compatibility)
    pub fn new_or_panic(lang: &str, model_path: &str) -> Self {
        Self::new(lang, model_path).unwrap_or_else(|e| {
            tracing::error!("Failed to create Phonemizer: {}", e);
            panic!("Failed to create Phonemizer: {}", e);
        })
    }

    /// Validate that a model path contains all required files
    /// Similar to how TTS model validates its path before loading
    /// This is a simple check - just verifies files exist (like TTS model validation)
    fn validate_model_path(path: &Path) -> Result<(), String> {
        let encoder_path = path.join("onnx").join("encoder_model.onnx");
        let decoder_path = path.join("onnx").join("decoder_model.onnx");
        let tokenizer_path = path.join("tokenizer.json");
        let vocab_path = path.join("vocab.json");
        let arpabet_mapping_path = path.join("arpabet-mapping.txt");

        // Check existence
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
            return Err(format!("ARPAbet mapping not found: {:?}", arpabet_mapping_path));
        }

        // Check accessibility (important for bundled files in production)
        for (name, file_path) in [
            ("encoder", &encoder_path),
            ("decoder", &decoder_path),
            ("tokenizer", &tokenizer_path),
            ("vocab", &vocab_path),
            ("arpabet mapping", &arpabet_mapping_path),
        ] {
            match std::fs::metadata(file_path) {
                Ok(metadata) => {
                    if metadata.len() == 0 {
                        return Err(format!("{} file is empty: {:?}", name, file_path));
                    }
                }
                Err(e) => {
                    return Err(format!(
                        "Cannot access {} file: {:?} - {}",
                        name, file_path, e
                    ));
                }
            }
        }

        tracing::debug!("✓ Validated G2P model path: {:?}", path);
        Ok(())
    }

    /// Find model path in common locations
    /// 
    /// **IMPROVEMENT**: This method now validates paths before returning them,
    /// ensuring files are accessible (important for production bundled apps).
    fn find_model_path() -> Option<PathBuf> {
        tracing::info!("Searching for mini-bart-g2p model...");

        // Check TAURI_RESOURCE_DIR first (most common in production)
        if let Ok(resource_dir) = std::env::var("TAURI_RESOURCE_DIR") {
            tracing::info!("  Checking TAURI_RESOURCE_DIR: {}", resource_dir);
            let path = PathBuf::from(&resource_dir).join("mini-bart-g2p");
            tracing::info!("    Checking path: {:?}", path);
            if path.exists() {
                tracing::info!("    Path exists: true");
                // Validate before returning
                if Self::validate_model_path(&path).is_ok() {
                    tracing::info!("    ✓ Found and validated model at: {:?}", path);
                    return Some(path);
                } else {
                    tracing::warn!("    ✗ Path exists but validation failed: {:?}", path);
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
                // Validate before returning
                if Self::validate_model_path(path).is_ok() {
                    tracing::info!("    ✓ Found and validated model at: {:?}", path);
                    return Some(path.clone());
                } else {
                    tracing::warn!("    ✗ Path exists but validation failed: {:?}", path);
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
