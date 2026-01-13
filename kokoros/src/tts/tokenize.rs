use crate::tts::vocab::VOCAB;

/// Tokenizes the given phonemes string into a vector of token indices.
///
/// This function takes a text string as input and converts it into a vector of token indices
/// by looking up each character in the global `VOCAB` map and mapping it to the corresponding
/// token index. The resulting vector contains the token indices for the input text.
///
/// # Arguments
/// * `text` - The input text string to be tokenized.
///
/// # Returns
/// A vector of `i64` token indices representing the input text.
pub fn tokenize(phonemes: &str) -> Vec<i64> {
    phonemes
        .chars()
        .filter_map(|c| VOCAB.get(&c))
        .map(|&idx| idx as i64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let text = "heɪ ðɪs ɪz ˈlʌvliː!";
        let tokens = tokenize(text);

        // Ensure tokenization yields some tokens
        assert!(!tokens.is_empty(), "Tokenization returned empty vector");
        // Verify that known characters are tokenized correctly
        // 'h' should map to its vocab index (lowercase h)
        let h_index = VOCAB.get(&'h').copied().expect("'h' not in VOCAB");
        assert!(tokens.contains(&h_index), "Tokens should contain index for 'h'");
        // Exclamation mark should map to its vocab index
        let excl_index = VOCAB.get(&'!').copied().expect("'!' not in VOCAB");
        assert!(tokens.contains(&excl_index), "Tokens should contain index for '!'" );
        // Test empty string
        let empty = "";
        let empty_tokens = tokenize(empty);
        assert!(empty_tokens.is_empty());
        // Test punctuation only
        let punct = "...";
        let punct_tokens = tokenize(punct);
        // All punctuation characters should be tokenized
        for ch in punct.chars() {
            let idx = VOCAB.get(&ch).copied().expect("punct char not in VOCAB");
            assert!(punct_tokens.contains(&idx), "Missing token for punctuation '{}'");
        }
    }
}

use crate::tts::vocab::REVERSE_VOCAB;

pub fn tokens_to_phonemes(tokens: &[i64]) -> String {
    tokens
        .iter()
        .filter_map(|&t| REVERSE_VOCAB.get(&(t as usize)))
        .collect()
}

#[cfg(test)]
mod tests2 {
    use super::*;

    #[test]
    fn test_tokens_to_phonemes() {
        let tokens = vec![24, 47, 54, 54, 57, 5];
        let text = tokens_to_phonemes(&tokens);
        assert_eq!(text, "Hello!");

        let tokens = vec![
            0, 50, 83, 54, 156, 57, 135, 3, 16, 65, 156, 87, 158, 54, 46, 5, 0,
        ];

        let text = tokens_to_phonemes(&tokens);
        assert_eq!(text, "$həlˈoʊ, wˈɜːld!$");

        // Test empty vector
        let empty_tokens: Vec<i64> = vec![];
        assert_eq!(tokens_to_phonemes(&empty_tokens), "");
    }
}
