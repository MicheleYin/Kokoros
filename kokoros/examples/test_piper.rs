// Simple test to verify Espeak backend loads and works
use kokoros::tts::phonemizer::Phonemizer;

fn main() {
    println!("Testing Espeak backend loading...\n");
    
    println!("Creating Phonemizer with Espeak backend...");
    let phonemizer = Phonemizer::new("a");
    
    println!("Backend type: {:?}", phonemizer.backend_type());
    println!("\nTesting phonemization:\n");
    
    let test_words = vec!["hello", "world", "test"];
    
    for word in test_words {
        let result = phonemizer.phonemize(word, false);
        match result.as_str() {
            "" => {
                println!("  '{}' -> (EMPTY - backend may not be working!)", word);
            }
            s => {
                println!("  '{}' -> '{}'", word, s);
            }
        }
    }
    
    if phonemizer.phonemize("hello", false).is_empty() {
        println!("\n⚠ WARNING: Espeak backend appears to not be working!");
        println!("   Check that espeak-rs is properly initialized.");
    } else {
        println!("\n✓ Espeak backend appears to be working!");
    }
}

