// Test the ByT5 8-layer model (g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx)
use kokoros::tts::phonemizer::Phonemizer;
use std::path::PathBuf;

fn main() {
    println!("=== Testing ByT5 8-Layer Model (g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx) ===\n");
    
    // Get model path from command line or use default
    let model_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .or_else(|| {
            // Try to find the 8-layer model in common locations
            let possible_paths = vec![
                PathBuf::from("../../src-tauri/resources/g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx"),
                PathBuf::from("../src-tauri/resources/g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx"),
                PathBuf::from("src-tauri/resources/g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx"),
            ];
            
            for path in possible_paths {
                if path.exists() && path.join("byt5_g2p_model.onnx").exists() {
                    println!("Found model at: {:?}\n", path);
                    return Some(path);
                }
            }
            
            println!("⚠ Using default path resolution (will search common locations)\n");
            None
        });
    
    println!("1. Creating Phonemizer (uses ByT5 backend by default)...");
    let phonemizer = Phonemizer::new("en");
    
    println!("   Backend type: {:?}", phonemizer.backend_type());
    println!("   Model path: {:?}\n", model_path);
    
    // Test 1: Basic phonemization (English)
    println!("2. Basic Phonemization (English):");
    let test_texts_en = vec![
        "hello",
        "world",
        "test",
        "example",
        "phoneme",
        "The quick brown fox",
    ];
    
    for text in &test_texts_en {
        let result = phonemizer.phonemize(text, false);
        if result.is_empty() {
            println!("   '{}' -> (EMPTY - model may not be loaded!)", text);
        } else {
            println!("   '{}' -> '{}'", text, result);
        }
    }
    println!();
    
    // Test 2: Multilingual support
    println!("3. Multilingual Phonemization:");
    println!("   Note: ByT5 model supports multiple languages via language tags");
    println!("   The phonemizer uses the language code from Phonemizer::new()\n");
    
    let test_texts_multilang = vec![
        ("hello", "en"),
        ("bonjour", "fr"),
        ("hola", "es"),
        ("guten tag", "de"),
    ];
    
    for (text, lang) in &test_texts_multilang {
        // Create phonemizer with specific language
        let phonemizer_lang = Phonemizer::new(lang);
        let result = phonemizer_lang.phonemize(text, false);
        if result.is_empty() {
            println!("   '{}' (lang: {}) -> (EMPTY)", text, lang);
        } else {
            println!("   '{}' (lang: {}) -> '{}'", text, lang, result);
        }
    }
    println!();
    
    // Test 3: Performance test
    println!("4. Performance Test (50 iterations):");
    let perf_word = "hello";
    let iterations = 50;
    
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = phonemizer.phonemize(perf_word, false);
    }
    let elapsed = start.elapsed();
    println!("   Time: {:.2}ms total ({:.2}μs per word)", 
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64
    );
    println!();
    
    // Test 4: Model status check
    println!("5. Model Status Check:");
    let test_result = phonemizer.phonemize("test", false);
    if test_result.is_empty() {
        println!("   ❌ Model appears to NOT be loaded or working!");
        println!("   Check that:");
        println!("     - model.onnx exists in model directory");
        println!("     - tokenizer_config.json exists in model directory");
        println!("\n   Note: This model uses tokenizer-free inference, so tokenizer.json is not needed.");
    } else {
        println!("   ✓ Model appears to be working!");
        println!("   ✓ Generated phonemes: '{}'", test_result);
    }
    println!();
    
    // Summary
    println!("Summary:");
    println!("  - ByT5 backend: Uses fdemelo/g2p-multilingual-byt5-tiny-8l-ipa-childes");
    println!("  - Model: g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx (8-layer)");
    println!("  - Supports: Multilingual G2P (31+ languages)");
    println!("  - Architecture: ByT5 (8 layers)");
    println!("  - Output: IPA phonemes directly");
    println!("  - Tokenizer-free inference (works with UTF-8 bytes)");
}

