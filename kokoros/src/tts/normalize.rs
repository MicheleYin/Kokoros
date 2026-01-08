use fancy_regex::Regex;
use lazy_static::lazy_static;

lazy_static! {
    static ref WHITESPACE_RE: Regex = Regex::new(r"[^\S \n]").unwrap();
    static ref MULTI_SPACE_RE: Regex = Regex::new(r"  +").unwrap();
    static ref NEWLINE_SPACE_RE: Regex = Regex::new(r"(?<=\n) +(?=\n)").unwrap();
    static ref DOCTOR_RE: Regex = Regex::new(r"\bD[Rr]\.(?= [A-Z])").unwrap();
    static ref MISTER_RE: Regex = Regex::new(r"\b(?:Mr\.|MR\.(?= [A-Z]))").unwrap();
    static ref MISS_RE: Regex = Regex::new(r"\b(?:Ms\.|MS\.(?= [A-Z]))").unwrap();
    static ref MRS_RE: Regex = Regex::new(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))").unwrap();
    static ref ETC_RE: Regex = Regex::new(r"\betc\.(?! [A-Z])").unwrap();
    static ref YEAH_RE: Regex = Regex::new(r"(?i)\b(y)eah?\b").unwrap();
    static ref NUMBERS_RE: Regex =
        Regex::new(r"\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)").unwrap();
    static ref COMMA_NUM_RE: Regex = Regex::new(r"(?<=\d),(?=\d)").unwrap();
    static ref MONEY_RE: Regex = Regex::new(
        r"(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b"
    )
    .unwrap();
    static ref POINT_NUM_RE: Regex = Regex::new(r"\d*\.\d+").unwrap();
    static ref RANGE_RE: Regex = Regex::new(r"(?<=\d)-(?=\d)").unwrap();
    static ref S_AFTER_NUM_RE: Regex = Regex::new(r"(?<=\d)S").unwrap();
    static ref POSSESSIVE_RE: Regex = Regex::new(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b").unwrap();
    static ref X_POSSESSIVE_RE: Regex = Regex::new(r"(?<=X')S\b").unwrap();
    static ref INITIALS_RE: Regex = Regex::new(r"(?:[A-Za-z]\.){2,} [a-z]").unwrap();
    static ref ACRONYM_RE: Regex = Regex::new(r"(?i)(?<=[A-Z])\.(?=[A-Z])").unwrap();
    // Match standalone integers (not part of words or already converted)
    static ref INTEGER_RE: Regex = Regex::new(r"\b\d+\b").unwrap();
    // Match Roman numerals (common in chapter titles)
    static ref ROMAN_NUM_RE: Regex = Regex::new(r"\b(?:I{1,3}|IV|VI{0,3}|IX|X{1,3}|XL|LX{0,3}|XC|C{1,3}|CD|DC{0,3}|CM|M{0,3})\b").unwrap();
}

pub fn normalize_text(text: &str) -> String {
    // Use catch_unwind to prevent panics from propagating - return original text if normalization panics
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut text = text.to_string();

        // Replace special quotes and brackets
        text = text.replace('\u{2018}', "'").replace('\u{2019}', "'");
        text = text.replace('«', "\u{201C}").replace('»', "\u{201D}");
        text = text.replace('\u{201C}', "\"").replace('\u{201D}', "\"");
        text = text.replace('(', "«").replace(')', "»");

        // Replace Chinese/Japanese punctuation
        let from_chars = ['、', '。', '！', '，', '：', '；', '？'];
        let to_chars = [',', '.', '!', ',', ':', ';', '?'];

        for (from, to) in from_chars.iter().zip(to_chars.iter()) {
            text = text.replace(*from, &format!("{} ", to));
        }

        // Apply regex replacements
        text = WHITESPACE_RE.replace_all(&text, " ").to_string();
        text = MULTI_SPACE_RE.replace_all(&text, " ").to_string();
        text = NEWLINE_SPACE_RE.replace_all(&text, "").to_string();
        text = DOCTOR_RE.replace_all(&text, "Doctor").to_string();
        text = MISTER_RE.replace_all(&text, "Mister").to_string();
        text = MISS_RE.replace_all(&text, "Miss").to_string();
        text = MRS_RE.replace_all(&text, "Mrs").to_string();
        text = ETC_RE.replace_all(&text, "etc").to_string();
        text = YEAH_RE.replace_all(&text, "${1}e'a").to_string();

        // Convert numbers to text before other number processing
        text = convert_numbers_to_text(&text);

        // Note: split_num, flip_money, and point_num functions need to be implemented
        text = COMMA_NUM_RE.replace_all(&text, "").to_string();
        text = RANGE_RE.replace_all(&text, " to ").to_string();
        text = S_AFTER_NUM_RE.replace_all(&text, " S").to_string();
        text = POSSESSIVE_RE.replace_all(&text, "'S").to_string();
        text = X_POSSESSIVE_RE.replace_all(&text, "s").to_string();

        // Handle initials and acronyms
        text = INITIALS_RE
            .replace_all(&text, |caps: &fancy_regex::Captures| {
                caps[0].replace('.', "-")
            })
            .to_string();
        text = ACRONYM_RE.replace_all(&text, "-").to_string();

        // Remove all non-alphabetical characters except basic punctuation symbols and whitespace
        // Keep: letters, numbers (for conversion), whitespace, and basic punctuation only
        // Basic punctuation: . , ! ? ; :
        text = text
            .chars()
            .filter(|c| {
                c.is_alphabetic() 
                || c.is_ascii_digit()  // Keep numbers so they can be converted to words
                || c.is_whitespace()
                || matches!(c, '.' | ',' | '!' | '?' | ';' | ':')
            })
            .collect();

        text.trim().to_string()
    }));

    // If normalization panicked, return original text (trimmed)
    result.unwrap_or_else(|_| text.trim().to_string())
}

/// Convert numbers and digits to their spelled textual representation
/// Examples: "123" -> "one hundred twenty three", "5" -> "five", "CHAPTER XIV" -> "CHAPTER fourteen"
/// This function is panic-safe - any panics in the conversion are caught and the original text is returned
fn convert_numbers_to_text(text: &str) -> String {
    // Use catch_unwind to prevent panics from propagating - use AssertUnwindSafe for the closure
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut result = text.to_string();

        // Convert Roman numerals first (common in chapter titles like "CHAPTER XIV")
        result = ROMAN_NUM_RE
            .replace_all(&result, |caps: &fancy_regex::Captures| {
                // Catch panics in the replacement callback - use AssertUnwindSafe for the closure
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let roman = caps[0].to_uppercase();
                    let num = roman_to_number(&roman);
                    if num > 0 {
                        number_to_words(num)
                    } else {
                        caps[0].to_string()
                    }
                }))
                .unwrap_or_else(|_| caps[0].to_string())
            })
            .to_string();

        // Convert Arabic numerals
        result = INTEGER_RE
            .replace_all(&result, |caps: &fancy_regex::Captures| {
                // Catch panics in the replacement callback - use AssertUnwindSafe for the closure
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if let Ok(num) = caps[0].parse::<u64>() {
                        number_to_words(num)
                    } else {
                        caps[0].to_string()
                    }
                }))
                .unwrap_or_else(|_| caps[0].to_string())
            })
            .to_string();

        result
    }));

    // If conversion panicked, return original text
    result.unwrap_or_else(|_| text.to_string())
}

/// Convert a number to its English word representation
fn number_to_words(n: u64) -> String {
    if n == 0 {
        return "zero".to_string();
    }

    const ONES: &[&str] = &[
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];

    const TENS: &[&str] = &[
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    fn convert_under_thousand(n: u64) -> String {
        if n == 0 {
            return String::new();
        }

        if n < 20 {
            return ONES[n as usize].to_string();
        }

        if n < 100 {
            let tens = (n / 10) as usize;
            let ones = (n % 10) as usize;
            if ones == 0 {
                TENS[tens].to_string()
            } else {
                format!("{} {}", TENS[tens], ONES[ones])
            }
        } else {
            let hundreds = (n / 100) as usize;
            let remainder = n % 100;
            if remainder == 0 {
                format!("{} hundred", ONES[hundreds])
            } else {
                format!(
                    "{} hundred {}",
                    ONES[hundreds],
                    convert_under_thousand(remainder)
                )
            }
        }
    }

    if n < 1000 {
        return convert_under_thousand(n);
    }

    let mut result = Vec::new();
    let mut remaining = n;

    const SCALES: &[(&str, u64)] = &[
        ("trillion", 1_000_000_000_000),
        ("billion", 1_000_000_000),
        ("million", 1_000_000),
        ("thousand", 1_000),
    ];

    for (scale_name, scale_value) in SCALES {
        if remaining >= *scale_value {
            let count = remaining / *scale_value;
            result.push(format!("{} {}", convert_under_thousand(count), scale_name));
            remaining %= *scale_value;
        }
    }

    if remaining > 0 {
        result.push(convert_under_thousand(remaining));
    }

    result.join(" ")
}

/// Convert Roman numerals to Arabic numbers
fn roman_to_number(roman: &str) -> u64 {
    let mut result = 0;
    let mut prev_value = 0;

    let values: std::collections::HashMap<char, u64> = [
        ('I', 1),
        ('V', 5),
        ('X', 10),
        ('L', 50),
        ('C', 100),
        ('D', 500),
        ('M', 1000),
    ]
    .iter()
    .cloned()
    .collect();

    for ch in roman.chars().rev() {
        let value = values.get(&ch).copied().unwrap_or(0);
        if value < prev_value {
            result -= value;
        } else {
            result += value;
        }
        prev_value = value;
    }

    result
}
