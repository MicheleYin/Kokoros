// Test comparing ByT5 model (g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx) against espeak-ng
use kokoros::tts::phonemizer::Phonemizer;
use std::process::Command;

fn get_espeak_phonemes(text: &str, lang: &str) -> Option<String> {
    // Try espeak-ng first, then espeak
    let espeak_cmd = if Command::new("espeak-ng").arg("--version").output().is_ok() {
        "espeak-ng"
    } else if Command::new("espeak").arg("--version").output().is_ok() {
        "espeak"
    } else {
        return None;
    };

    // Map language codes
    let espeak_lang = match lang {
        "en" | "a" => "en",
        "en-us" | "en-US" => "en-us",
        "en-gb" | "en-GB" | "b" => "en-gb",
        _ => lang,
    };

    // Use espeak-ng/espeak to get IPA phonemes
    // -x outputs phoneme names, --ipa outputs IPA
    let output = Command::new(espeak_cmd)
        .args(&["-x", "--ipa", "-v", espeak_lang, text])
        .output()
        .ok()?;

    if output.status.success() {
        let phonemes = String::from_utf8_lossy(&output.stdout);
        Some(phonemes.trim().to_string())
    } else {
        None
    }
}

fn main() {
    println!("=== Comparing ByT5 Model vs espeak-ng ===\n");

    // Check if espeak is available
    let espeak_available = Command::new("espeak-ng").arg("--version").output().is_ok()
        || Command::new("espeak").arg("--version").output().is_ok();
    
    if !espeak_available {
        println!("⚠ espeak-ng/espeak not found in PATH");
        println!("  Install with: brew install espeak-ng (macOS) or apt-get install espeak-ng (Linux)");
        println!("  Continuing with ByT5-only tests...\n");
    }

    // Test cases
    let test_cases = vec![
        ("hello", "en"),
        ("world", "en"),
        ("test", "en"),
        ("example", "en"),
        ("phoneme", "en"),
        ("The quick brown fox", "en"),
        ("Hello, how are you today?", "en"),
        ("bonjour", "fr"),
        ("hola", "es"),
        ("guten tag", "de"),
    ];

    println!("1. Phoneme Comparison Table:");
    println!("   {:<30} | {:<50} | {:<50}", "Text (lang)", "ByT5 Model", "espeak-ng");
    println!("   {}", "-".repeat(135));

    let _byt5_phonemizer = Phonemizer::new("en");
    let mut matches = 0;
    let mut total = 0;

    for (text, lang) in &test_cases {
        total += 1;
        
        // Get ByT5 output
        let byt5_phonemizer_lang = Phonemizer::new(lang);
        let byt5_output = byt5_phonemizer_lang.phonemize(text, false);
        let byt5_display = if byt5_output.is_empty() {
            "(EMPTY)".to_string()
        } else {
            byt5_output.clone()
        };

        // Get espeak output
        let espeak_output = if espeak_available {
            get_espeak_phonemes(text, lang).unwrap_or_else(|| "(ERROR)".to_string())
        } else {
            "(N/A)".to_string()
        };

        // Compare (simple string comparison for now)
        let is_similar = if espeak_available && !byt5_output.is_empty() {
            // Normalize both outputs for comparison (remove spaces, convert to lowercase)
            let byt5_norm: String = byt5_output.chars().filter(|c| !c.is_whitespace()).collect();
            let espeak_norm: String = espeak_output.chars().filter(|c| !c.is_whitespace()).collect();
            
            // Check if they're similar (at least 50% character overlap)
            let common_chars: usize = byt5_norm.chars()
                .zip(espeak_norm.chars())
                .filter(|(a, b)| a == b)
                .count();
            let similarity = if byt5_norm.len().max(espeak_norm.len()) > 0 {
                (common_chars as f64) / (byt5_norm.len().max(espeak_norm.len()) as f64)
            } else {
                0.0
            };
            
            if similarity > 0.5 {
                matches += 1;
                true
            } else {
                false
            }
        } else {
            false
        };

        let marker = if is_similar { "✓" } else { " " };
        println!("   {:<30} | {:<50} | {:<50} {}", 
            format!("{} ({})", text, lang), 
            byt5_display, 
            espeak_output,
            marker
        );
    }

    println!("\n2. Summary:");
    if espeak_available {
        println!("   Total test cases: {}", total);
        println!("   Similar outputs: {} ({:.1}%)", matches, (matches as f64 / total as f64) * 100.0);
        println!("   Note: 'Similar' means >50% character overlap after normalization");
    } else {
        println!("   espeak-ng not available - only ByT5 outputs shown");
    }
    println!();

    // Test 3: Performance comparison
    println!("3. Performance Comparison (100 iterations):");
    let perf_word = "hello";
    let iterations = 100;

    // ByT5
    let byt5_perf_phonemizer = Phonemizer::new("en");
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = byt5_perf_phonemizer.phonemize(perf_word, false);
    }
    let byt5_time = start.elapsed();
    println!("   ByT5 Model:      {:.2}ms ({:.2}μs per word)", 
        byt5_time.as_secs_f64() * 1000.0,
        byt5_time.as_secs_f64() * 1_000_000.0 / iterations as f64
    );

    // espeak
    if espeak_available {
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = get_espeak_phonemes(perf_word, "en");
        }
        let espeak_time = start.elapsed();
        println!("   espeak-ng:       {:.2}ms ({:.2}μs per word)", 
            espeak_time.as_secs_f64() * 1000.0,
            espeak_time.as_secs_f64() * 1_000_000.0 / iterations as f64
        );
        println!("   Speedup:         {:.2}x", espeak_time.as_secs_f64() / byt5_time.as_secs_f64());
    } else {
        println!("   espeak-ng:       (N/A - not installed)");
    }
    println!();

    // Test 4: Detailed comparison for a few examples
    println!("4. Detailed Comparison (selected examples):");
    let detailed_tests = vec![
        ("hello", "en"),
        ("world", "en"),
        ("The quick brown fox", "en"),
    ];

    for (text, lang) in &detailed_tests {
        println!("\n   Text: '{}' (lang: {})", text, lang);
        
        let byt5_phonemizer_lang = Phonemizer::new(lang);
        let byt5_output = byt5_phonemizer_lang.phonemize(text, false);
        println!("   ByT5:  {}", if byt5_output.is_empty() { "(EMPTY)" } else { &byt5_output });

        if espeak_available {
            let espeak_output = get_espeak_phonemes(text, lang).unwrap_or_else(|| "(ERROR)".to_string());
            println!("   espeak: {}", espeak_output);
            
            // Show differences
            if !byt5_output.is_empty() && espeak_output != "(ERROR)" {
                let byt5_chars: Vec<char> = byt5_output.chars().filter(|c| !c.is_whitespace()).collect();
                let espeak_chars: Vec<char> = espeak_output.chars().filter(|c| !c.is_whitespace()).collect();
                
                let min_len = byt5_chars.len().min(espeak_chars.len());
                let max_len = byt5_chars.len().max(espeak_chars.len());
                let diff_count = byt5_chars.iter()
                    .zip(espeak_chars.iter())
                    .filter(|(a, b)| a != b)
                    .count() + (max_len - min_len);
                
                println!("   Diff:  {} characters differ (length: {} vs {})", 
                    diff_count, byt5_chars.len(), espeak_chars.len());
            }
        }
    }
    println!();

    // Test 5: Model information
    println!("5. Model Information:");
    println!("   ByT5 Model:");
    println!("     - Model: g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx");
    println!("     - Architecture: ByT5 (8 layers)");
    println!("     - Languages: Multilingual (31+ languages)");
    println!("     - Output: IPA phonemes directly");
    println!("     - Tokenizer-free inference");
    println!();
    println!("   espeak-ng:");
    println!("     - Rule-based phonemization");
    println!("     - Supports 100+ languages");
    println!("     - Output: IPA phonemes");
    println!("     - Fast, deterministic");
    println!();

    println!("Summary:");
    println!("  - ByT5: Neural model, learns from data, may have better quality for some words");
    println!("  - espeak-ng: Rule-based, fast, consistent, good coverage");
    println!("  - Both output IPA phonemes, but may differ in representation");
}

