// Simple test to verify piper-tts-rust backend loads and works
use kokoros::tts::phonemizer::{BackendType, Phonemizer};
use std::path::PathBuf;

fn main() {
    println!("Testing piper-tts-rust backend loading...\n");
    
    // Get model path from command line or use default
    let model_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("../../mini-bart-g2p")));
    
    println!("Creating Phonemizer with piper-tts-rust backend...");
    let phonemizer = Phonemizer::new_with_backend("a", BackendType::PiperTtsRust, model_path);
    
    println!("Backend type: {:?}", phonemizer.backend_type());
    println!("\nTesting phonemization:\n");
    
    let test_words = vec!["hello", "world", "test"];
    
    for word in test_words {
        let result = phonemizer.phonemize(word, false);
        match result.as_str() {
            "" => {
                println!("  '{}' -> (EMPTY - model may not be loaded!)", word);
            }
            s => {
                println!("  '{}' -> '{}'", word, s);
            }
        }
    }
    
    if phonemizer.phonemize("hello", false).is_empty() {
        println!("\n⚠ WARNING: Model appears to not be loaded or working!");
        println!("   Check that model files exist at the specified path.");
    } else {
        println!("\n✓ Model appears to be working!");
    }
}

