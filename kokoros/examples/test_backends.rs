// Example: Test both Espeak and RuleBasedG2p backends
use kokoros::tts::phonemizer::{BackendType, Phonemizer};

fn main() {
    println!("=== Testing Phonemizer Backends ===\n");

    let test_text = "hello world";

    // Test 1: Espeak Backend
    println!("1. Testing Espeak Backend:");
    println!("   Creating phonemizer with lang='a' (en-us)...");
    let espeak_phonemizer = Phonemizer::new("a");
    println!("   ✓ Espeak phonemizer created successfully");
    println!("   Backend type: {:?}", espeak_phonemizer.backend_type());
    
    // Test phonemization
    let phonemes = espeak_phonemizer.phonemize(test_text, false);
    if !phonemes.is_empty() {
        println!("   Input:  '{}'", test_text);
        println!("   Output: '{}'", phonemes);
        println!("   ✓ Phonemization successful\n");
    } else {
        println!("   ⚠ Empty phonemes\n");
    }

    // Test 2: RuleBasedG2p Backend (voirs-g2p)
    println!("2. Testing RuleBasedG2p Backend (voirs-g2p):");
    let test_languages = vec!["a", "b", "en-us", "en-gb", "en"];

    for lang in test_languages {
        println!("   Testing language: {}", lang);
        let phonemizer = Phonemizer::new_with_g2p(lang, None);
        
        if phonemizer.backend_type() == BackendType::VoirsG2p {
            println!("   ✓ RuleBasedG2p phonemizer created successfully");
            
            // Test phonemization (without normalization to avoid regex issues)
            let phonemes = phonemizer.phonemize(test_text, false);
            if !phonemes.is_empty() {
                println!("   Input:  '{}'", test_text);
                println!("   Output: '{}'", phonemes);
                println!("   Note: RuleBasedG2p uses rule-based phonemization");
                println!("         Outputs may differ from espeak's IPA format.\n");
            } else {
                println!("   ⚠ Empty phonemes (language may not be supported)\n");
            }
        } else {
            println!("   ⚠ Language not supported, fell back to Espeak\n");
        }
    }

    // Test 3: Explicit backend selection
    println!("3. Testing Explicit Backend Selection:");
    
    // Test with explicit Espeak
    let espeak = Phonemizer::new_with_backend("a", BackendType::Espeak, None);
    println!("   Espeak backend: {:?}", espeak.backend_type());
    
    // Test with explicit RuleBasedG2p
    let g2p = Phonemizer::new_with_backend("a", BackendType::VoirsG2p, None);
    println!("   RuleBasedG2p backend: {:?}", g2p.backend_type());
    println!("   Note: RuleBasedG2p doesn't require model files\n");

    println!("\n=== Backend Testing Complete ===");
}

