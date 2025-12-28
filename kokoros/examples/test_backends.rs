// Example: Test piper-tts-rust backend
use kokoros::tts::phonemizer::{BackendType, Phonemizer};
use std::path::PathBuf;

fn main() {
    println!("=== Testing piper-tts-rust Backend ===\n");

    let test_texts = vec![
        "hello",
        "world",
        "the",
        "quick",
        "brown",
        "fox",
        "The quick brown fox",
        "Hello, how are you today?",
    ];

    // Get model path from command line or use default
    let model_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("../../mini-bart-g2p")));

    // Test 1: Basic phonemization
    println!("1. Basic Phonemization (American English):");
    println!("   Language: 'a' (en-us)\n");
    
    for text in &test_texts {
        println!("   Text: '{}'", text);
        
        // piper-tts-rust
        let piper_tts = Phonemizer::new_with_backend("a", BackendType::PiperTtsRust, model_path.clone());
        let piper_tts_output = piper_tts.phonemize(text, false);
        println!("   piper-tts-rust:   '{}'", piper_tts_output);
        println!();
    }

    // Test 2: Word-by-word comparison
    println!("2. Word-by-Word Phonemization:");
    println!("   {:<15} | {:<35}", "Word", "piper-tts-rust");
    println!("   {}", "-".repeat(55));
    
    let test_words = vec!["hello", "world", "the", "quick", "brown", "fox", "today", "are", "you", "test"];
    
    for word in &test_words {
        let piper_tts = Phonemizer::new_with_backend("a", BackendType::PiperTtsRust, model_path.clone());
        let piper_tts_out = piper_tts.phonemize(word, false);
        
        println!("   {:<15} | {:<35}", word, &piper_tts_out);
    }
    println!();

    // Test 3: Backend information
    println!("3. Backend Information:");
    println!("   piper-tts-rust:");
    println!("     - Uses piper-tts-rust library");
    println!("     - ONNX-based neural G2P model (mini-bart-g2p)");
    println!("     - BART seq2seq architecture");
    println!("     - Converts ARPABET to IPA");
    println!("     - Requires ONNX model files and arpabet-mapping.txt");
    println!();

    // Test 4: Performance test
    println!("4. Performance Test (100 iterations):");
    let perf_word = "hello";
    let iterations = 100;
    
    // piper-tts-rust
    let piper_tts = Phonemizer::new_with_backend("a", BackendType::PiperTtsRust, model_path.clone());
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = piper_tts.phonemize(perf_word, false);
    }
    let piper_tts_time = start.elapsed();
    println!("   piper-tts-rust: {:.2}ms ({:.2}μs per word)", 
        piper_tts_time.as_secs_f64() * 1000.0,
        piper_tts_time.as_secs_f64() * 1_000_000.0 / iterations as f64
    );
    println!();

    println!("Summary:");
    println!("  - piper-tts-rust: Neural ONNX-based G2P model using mini-bart-g2p");
    println!("  - Provides high-quality phonemization for TTS");
}
