use kokoros::tts::phonemizer::Phonemizer;
use std::env;

fn main() {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    println!("=== Testing ONNX Phonemizer (mini-bart-g2p) ===\n");

    // Check environment variables
    println!("Environment check:");
    match env::var("TAURI_RESOURCE_DIR") {
        Ok(val) => println!("  TAURI_RESOURCE_DIR: {}", val),
        Err(_) => println!("  TAURI_RESOURCE_DIR: not set"),
    }
    match env::current_dir() {
        Ok(dir) => println!("  Current directory: {:?}", dir),
        Err(e) => println!("  Current directory: error: {}", e),
    }
    println!();

    // Try to create phonemizer
    println!("1. Creating Phonemizer...");
    let phonemizer = match Phonemizer::new("en") {
        Ok(p) => {
            println!("   ✓ Phonemizer created successfully");
            println!("   Backend type: {:?}", p.backend_type());
            p
        }
        Err(e) => {
            eprintln!("   ✗ Failed to create Phonemizer: {}", e);
            eprintln!("\nTroubleshooting:");
            eprintln!("   - Make sure TAURI_RESOURCE_DIR is set to the resources directory");
            eprintln!("   - Or ensure mini-bart-g2p model is in one of these locations:");
            eprintln!("     * src-tauri/resources/mini-bart-g2p");
            eprintln!("     * resources/mini-bart-g2p");
            eprintln!("     * ../src-tauri/resources/mini-bart-g2p");
            eprintln!("     * ../../src-tauri/resources/mini-bart-g2p");
            std::process::exit(1);
        }
    };

    // Test cases including edge cases with UTF-8, numbers, and special characters
    // Store long text in a variable first to avoid temporary value issues
    let long_text = "a".repeat(1000);

    // Store problematic Project Gutenberg text that causes "No tokens generated" error
    let problematic_gutenberg_text = "Title: Little Women Author: Louisa May Alcott Release date: May 1, 1996 [eBook #514] Most recently updated: November 4, 2022 Language: English *** START OF THE PROJECT GUTENBERG EBOOK LITTLE WOMEN *** Little Women by Louisa May Alcott Contents PART 1 CHAPTER ONE PLAYING PILGRIMS CHAPTER TWO A MERRY CHRISTMAS CHAPTER THREE THE LAURENCE BOY CHAPTER FOUR BURDENS CHAPTER FIVE BEING NEIGHBORLY CHAPTER SIX BETH FINDS THE PALACE BEAUTIFUL CHAPTER SEVEN AMY'S VALLEY OF HUMILIATION.".to_string();

    // Store another problematic 10-word chunk that causes "No tokens generated" error
    // This is a chunk from the split text that still fails even at 10 words
    let problematic_chunk_text =
        "language: english *** start of the project gutenberg ebook little.".to_string();

    let mut test_cases: Vec<&str> = vec![
        "hello",
        "world",
        "the quick brown fox",
        "testing phonemization",
        "Hello, world!",
        "123",
        "",
        // Number edge cases
        "CHAPTER XIV",
        "CHAPTER 14",
        "CHAPTER 123",
        "I have 5 apples and 42 oranges",
        "The year 2024",
        "1234567890",
        "CHAPTER I",
        "CHAPTER II",
        "CHAPTER III",
        "CHAPTER IV",
        "CHAPTER V",
        "CHAPTER X",
        "CHAPTER XX",
        "CHAPTER XXX",
        // UTF-8 edge cases
        "café",
        "naïve",
        "résumé",
        "Zürich",
        "São Paulo",
        "Müller",
        "北京",
        "こんにちは",
        "Здравствуй",
        "مرحبا",
        "🎉🎊🎈",
        "Hello—world", // em dash
        "Hello–world", // en dash
        "Hello…world", // ellipsis
        "\"quoted text\"",
        "'single quotes'",
        "«French quotes»",
        "„German quotes„",
        "「Japanese quotes」",
        // Special punctuation
        "Dr. Smith",
        "Mr. Jones",
        "Mrs. Brown",
        "Ms. Davis",
        "etc.",
        "U.S.A.",
        "Ph.D.",
        "A.I.",
        "NASA",
        "FBI",
        // Mixed content
        "In 2024, CHAPTER XIV had 42 pages.",
        "The price is $123.45",
        "Temperature: -5°C",
        "Score: 100%",
        "Version 2.0",
        "3.14159",
        // Empty and whitespace
        "   ",
        "\n\n",
        "\t\t",
        "\r\n",
        // Control characters (should be filtered)
        "\x00\x01\x02",
        // Mixed scripts
        "Hello 世界",
        "123中文",
        "English123中文",
        // Zero-width characters
        "hello\u{200B}world", // zero-width space
        "hello\u{200C}world", // zero-width non-joiner
        "hello\u{200D}world", // zero-width joiner
        // Combining characters
        "caf\u{00E9}",  // é as combining character
        "na\u{00EF}ve", // ï as combining character
    ];

    // Add very long text to test cases
    test_cases.push(&long_text);

    // Add problematic Project Gutenberg text that causes "No tokens generated" error
    // This text contains special characters, multiple asterisks, and formatting that may cause issues
    test_cases.push(&problematic_gutenberg_text);

    // Add problematic 10-word chunk that still causes failures
    // This chunk contains asterisks and special formatting that causes phonemizer issues
    test_cases.push(&problematic_chunk_text);

    println!("\n2. Testing phonemization:");
    println!(
        "   Note: Testing problematic Project Gutenberg text that previously caused 'No tokens generated' error"
    );
    println!("   Total test cases: {}\n", test_cases.len());

    let mut passed = 0;
    let mut failed = 0;
    let mut empty_results = 0;

    for (i, text) in test_cases.iter().enumerate() {
        // Show text representation (escape special chars for display)
        let display_text = if text.len() > 50 {
            format!("{}... ({} chars)", &text[..50], text.len())
        } else {
            text.replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t")
                .replace('\x00', "\\0")
        };

        print!("   Test {}: '{}'", i + 1, display_text);

        let phonemes = phonemizer.phonemize(text, true);

        if phonemes.is_empty() {
            if text.trim().is_empty() {
                println!(" → ✓ EMPTY (expected for empty input)");
                passed += 1;
            } else {
                println!(" → ✗ EMPTY (unexpected!)");
                println!(
                    "       Input length: {} chars, UTF-8 bytes: {}",
                    text.chars().count(),
                    text.len()
                );
                println!(
                    "       First 10 chars: {:?}",
                    text.chars().take(10).collect::<Vec<_>>()
                );
                failed += 1;
                empty_results += 1;
            }
        } else {
            println!(" → ✓ '{}'", phonemes);
            println!("       Length: {} characters", phonemes.len());
            passed += 1;
        }
    }

    println!(
        "\n   Summary: {} passed, {} failed ({} empty results)",
        passed, failed, empty_results
    );

    println!("\n3. Testing with normalization disabled:");
    let test_text = "Hello World";
    let with_norm = phonemizer.phonemize(test_text, true);
    let without_norm = phonemizer.phonemize(test_text, false);
    println!("   Text: '{}'", test_text);
    println!("   With normalization: '{}'", with_norm);
    println!("   Without normalization: '{}'", without_norm);

    println!("\n=== Test Complete ===");
}
