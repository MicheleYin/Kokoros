// Debug example to see raw RuleBasedG2p output
use kokoros::tts::phonemizer::{BackendType, Phonemizer};

fn main() {
    // Enable debug logging
    std::env::set_var("RUST_LOG", "debug");
    // Note: env_logger is optional - remove this line if the crate is not available
    // env_logger::init();
    
    println!("=== Debugging RuleBasedG2p Output ===\n");
    
    // Test with a simple word
    let test_words = vec!["hello", "world", "test"];
    
    for word in test_words {
        println!("Testing word: '{}'", word);
        let phonemizer = Phonemizer::new_with_g2p("a", None);
        
        if phonemizer.backend_type() == BackendType::VoirsG2p {
            let phonemes = phonemizer.phonemize(word, false);
            println!("  Output: '{}'", phonemes);
            println!("  Length: {}", phonemes.len());
            println!("  Chars: {:?}\n", phonemes.chars().take(20).collect::<Vec<_>>());
        } else {
            println!("  ⚠ RuleBasedG2p backend not active\n");
        }
    }
}

