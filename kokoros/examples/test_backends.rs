// Example: Test Espeak G2P backend
use kokoros::tts::phonemizer::Phonemizer;

fn main() {
    println!("=== Testing Espeak G2P Backend ===\n");

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

    // Test 1: Basic phonemization
    println!("1. Basic Phonemization (American English):");
    println!("   Language: 'a' (en-us)\n");
    
    let phonemizer = Phonemizer::new("a");
    
    for text in &test_texts {
        println!("   Text: '{}'", text);
        let output = phonemizer.phonemize(text, false);
        println!("   Espeak: '{}'", output);
        println!();
    }

    // Test 2: Word-by-word phonemization
    println!("2. Word-by-Word Phonemization:");
    println!("   {:<15} | {:<50}", "Word", "Espeak");
    println!("   {}", "-".repeat(70));
    
    let test_words = vec!["hello", "world", "the", "quick", "brown", "fox", "today", "are", "you", "test"];
    
    for word in &test_words {
        let output = phonemizer.phonemize(word, false);
        println!("   {:<15} | {:<50}", word, &output);
    }
    println!();

    // Test 3: Multilingual support
    println!("3. Multilingual Support:");
    println!("   Note: Espeak supports multiple languages\n");
    
    let multilingual_tests = vec![
        ("hello", "en"),
        ("bonjour", "fr"),
        ("hola", "es"),
        ("guten tag", "de"),
    ];
    
    for (text, lang) in &multilingual_tests {
        let lang_phonemizer = Phonemizer::new(lang);
        let output = lang_phonemizer.phonemize(text, false);
        println!("   '{}' (lang: {}) -> '{}'", text, lang, output);
    }
    println!();

    // Test 4: Backend information
    println!("4. Backend Information:");
    println!("   Espeak:");
    println!("     - Uses piper-rs espeak-rs crate");
    println!("     - Rule-based phonemization");
    println!("     - Supports 100+ languages");
    println!("     - Output: IPA phonemes directly");
    println!("     - Fast, deterministic");
    println!();

    // Test 5: Performance test
    println!("5. Performance Test (100 iterations):");
    let perf_word = "hello";
    let iterations = 100;
    
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = phonemizer.phonemize(perf_word, false);
    }
    let espeak_time = start.elapsed();
    println!("   Espeak:          {:.2}ms ({:.2}μs per word)", 
        espeak_time.as_secs_f64() * 1000.0,
        espeak_time.as_secs_f64() * 1_000_000.0 / iterations as f64
    );
    println!();

    // Test 6: Status check
    println!("6. Backend Status Check:");
    let test_output = phonemizer.phonemize("test", false);
    
    if test_output.is_empty() {
        println!("   ❌ Espeak backend appears to NOT be working!");
    } else {
        println!("   ✓ Espeak backend is working! Generated: '{}'", test_output);
    }
    println!();

    println!("Summary:");
    println!("  - Espeak: Rule-based phonemization using piper-rs espeak-rs crate");
    println!("    • Supports 100+ languages");
    println!("    • Fast and deterministic");
    println!("    • Direct IPA output");
    println!("    • No model files required");
}
