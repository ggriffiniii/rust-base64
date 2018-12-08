//! # Configs
//!
//! There isn't just one type of Base64; that would be too simple. You need to choose a character
//! set (standard, URL-safe, etc) and padding suffix (yes/no).
//! The `Config` struct encapsulates this info. There are some common configs included: `STANDARD`,
//! `URL_SAFE`, etc. You can also make your own `Config` if needed.
//!
//! The functions that don't have `config` in the name (e.g. `encode()` and `decode()`) use the
//! `STANDARD` config .
//!
//! The functions that write to a slice (the ones that end in `_slice`) are generally the fastest
//! because they don't need to resize anything. If it fits in your workflow and you care about
//! performance, keep using the same buffer (growing as need be) and use the `_slice` methods for
//! the best performance.
//!
//! # Encoding
//!
//! Several different encoding functions are available to you depending on your desire for
//! convenience vs performance.
//!
//! | Function                | Output                       | Allocates                      |
//! | ----------------------- | ---------------------------- | ------------------------------ |
//! | `encode`                | Returns a new `String`       | Always                         |
//! | `encode_config`         | Returns a new `String`       | Always                         |
//! | `encode_config_buf`     | Appends to provided `String` | Only if `String` needs to grow |
//! | `encode_config_slice`   | Writes to provided `&[u8]`   | Never                          |
//!
//! All of the encoding functions that take a `Config` will pad as per the config.
//!
//! # Decoding
//!
//! Just as for encoding, there are different decoding functions available.
//!
//! | Function                | Output                        | Allocates                      |
//! | ----------------------- | ----------------------------- | ------------------------------ |
//! | `decode`                | Returns a new `Vec<u8>`       | Always                         |
//! | `decode_config`         | Returns a new `Vec<u8>`       | Always                         |
//! | `decode_config_buf`     | Appends to provided `Vec<u8>` | Only if `Vec` needs to grow    |
//! | `decode_config_slice`   | Writes to provided `&[u8]`    | Never                          |
//!
//! Unlike encoding, where all possible input is valid, decoding can fail (see `DecodeError`).
//!
//! Input can be invalid because it has invalid characters or invalid padding. (No padding at all is
//! valid, but excess padding is not.) Whitespace in the input is invalid.
//!
//! # Panics
//!
//! If length calculations result in overflowing `usize`, a panic will result.
//!
//! The `_slice` flavors of encode or decode will panic if the provided output slice is too small,

#![cfg_attr(feature = "cargo-clippy", allow(cast_lossless))]
#![deny(
    // TODO: Reenable missing_docs
    // missing_docs,
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_import_braces,
    unused_results,
    variant_size_differences,
    warnings,
)]

extern crate byteorder;
#[macro_use]
extern crate cfg_if;

mod chunked_encoder;
pub mod display;
mod tables;
pub mod write;

mod encode;
pub use encode::{encode, encode_config, encode_config_buf, encode_config_slice, Encoding};

mod decode;
pub use decode::{decode, decode_config, decode_config_buf, decode_config_slice, DecodeError, Decoding};

#[cfg(test)]
mod tests;

pub type Standard = Config<StandardAlphabet, WithPadding>;
pub type StandardNoPad = Config<StandardAlphabet, NoPadding>;
pub type UrlSafe = Config<UrlSafeAlphabet, WithPadding>;
pub type UrlSafeNoPad = Config<UrlSafeAlphabet, NoPadding>;
pub type Crypt = Config<CryptAlphabet, WithPadding>;
pub type CryptNoPad = Config<CryptAlphabet, NoPadding>;

pub const STANDARD: Standard = Config(StandardAlphabet, WithPadding);
pub const STANDARD_NO_PAD: StandardNoPad = Config(StandardAlphabet, NoPadding);
pub const URL_SAFE: UrlSafe = Config(UrlSafeAlphabet, WithPadding);
pub const URL_SAFE_NO_PAD: UrlSafeNoPad = Config(UrlSafeAlphabet, NoPadding);
pub const CRYPT: Crypt = Config(CryptAlphabet, WithPadding);
pub const CRYPT_NO_PAD: CryptNoPad = Config(CryptAlphabet, NoPadding);

// Module for trait sealing. The configuration traits are part of the public API
// because public functions (e.g. encode_config, decode_config, etc.) are
// bounded by them, but (atleast for now) we don't intend for outside
// crates to implement them. This provides more flexibility if we decide to
// change some of the traits behaviors. By having all the traits require
// private::Sealed to be implemented, we can effectively enforce that nobody
// outside this crate can implement the trait because the `private` module is
// not publicly accessible.
mod private {
    pub trait Sealed {}
}

/// Padding defines whether padding is used when encoding/decoding and if so
/// which character is to be used.
pub trait Padding : private::Sealed + Copy {
    fn padding_byte(self) -> Option<u8>;

    #[inline]
    fn has_padding(self) -> bool {
        self.padding_byte().is_some()
    }
}

/// WithPadding specifies to use the standard padding character b'='.
#[derive(Debug, Default, Clone, Copy)]
pub struct WithPadding;
impl Padding for WithPadding {
    #[inline]
    fn padding_byte(self) -> Option<u8> { Some(b'=') }
}
impl private::Sealed for WithPadding {}

/// NoPadding specifies that no padding is used when encoding/decoding.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoPadding;
impl Padding for NoPadding {
    #[inline]
    fn padding_byte(self) -> Option<u8> { None }
}
impl private::Sealed for NoPadding {}

/// Config wraps an `alphabet` (a type that implements Encoding+Decoding) and a
/// Padding. These are the basic requirements to use all the publicly accessible
/// functions.
#[derive(Debug, Default, Clone, Copy)]
pub struct Config<A: Copy, P: Copy>(A, P);

impl<A, P> Padding for Config<A, P> where A: Copy, P: Padding {
    #[inline]
    fn padding_byte(self) -> Option<u8> {
        self.1.padding_byte()
    }
}

impl<A, P> encode::Encoding for Config<A, P> where A: encode::Encoding, P: Copy {
    #[inline]
    fn encode_u6(self, input: u8) -> u8 {
        self.0.encode_u6(input)
    }
}

impl<A, P> encode::bulk_encoding::IntoBulkEncoding for Config<A, P> where A: encode::bulk_encoding::IntoBulkEncoding, P: Copy {
    type BulkEncoding = A::BulkEncoding;

    #[inline]
    fn into_bulk_encoding(self) -> Self::BulkEncoding {
        self.0.into_bulk_encoding()
    }
}

impl<A, P> decode::Decoding for Config<A, P> where A: decode::Decoding, P: Copy {
    #[inline]
    fn decode_u8(self, input: u8) -> u8 {
        self.0.decode_u8(input)
    }

    #[inline]
    fn invalid_value(self) -> u8 {
        self.0.invalid_value()
    }
}

impl<A, P> private::Sealed for Config<A, P> where A: Copy, P: Copy {}

#[derive(Debug, Default, Clone, Copy)]
pub struct StandardAlphabet;

#[derive(Debug, Default, Clone, Copy)]
pub struct UrlSafeAlphabet;

#[derive(Debug, Default, Clone, Copy)]
pub struct CryptAlphabet;

impl ::private::Sealed for StandardAlphabet {}
impl ::private::Sealed for UrlSafeAlphabet {}
impl ::private::Sealed for CryptAlphabet {}

#[derive(Clone)]
pub struct ConfigBuilder{
    alphabet: [u8;64],
    padding_byte: Option<u8>,
}

impl std::fmt::Debug for ConfigBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ConfigBuilder{{alphabet: {:?}, padding_byte: {:?}}}", &self.alphabet[..], self.padding_byte)
    }
}


impl ConfigBuilder {
    pub fn with_alphabet(alphabet: [u8;64]) -> ConfigBuilder {
        ConfigBuilder{
            alphabet,
            padding_byte: Some(b'='),
        }
    }

    pub fn with_padding(mut self, padding_byte: u8) -> ConfigBuilder {
        self.padding_byte = Some(padding_byte);
        self
    }

    pub fn no_padding(mut self) -> ConfigBuilder {
        self.padding_byte = None;
        self
    }

    // TODO: Use something better than a String for error.
    pub fn build(self) -> Result<CustomConfig, String> {
        let mut decode_scratch: Vec<Option<u8>> = vec![None;256];
        for (i, b) in self.alphabet.iter().cloned().enumerate() {
            if decode_scratch[b as usize].is_some() {
                return Err(format!("Duplicate value in alphabet: {}", b));
            }
            decode_scratch[b as usize] = Some(i as u8);
        }
        // One of the unused values is used as a sentinel `invalid_value`. Arbitrarily use the lowest unused byte.
        let invalid_value = decode_scratch.iter().enumerate().filter(|(_, b)| b.is_none()).nth(0).map(|(i, _)| i as u8).expect("should always have atleast one unused value in the decode table");
        let decode_scratch: Vec<u8> = decode_scratch.into_iter().map(|b| b.unwrap_or(invalid_value)).collect();
        let mut decode_table = [0;256];
        decode_table.copy_from_slice(&decode_scratch);
        Ok(CustomConfig{
            encode_table: self.alphabet,
            decode_table: decode_table,
            invalid_value: invalid_value,
            padding_byte: self.padding_byte,
        })
    }
}

#[derive(Clone)]
pub struct CustomConfig{
    encode_table: [u8;64],
    decode_table: [u8;256],
    invalid_value: u8,
    padding_byte: Option<u8>,
}

impl std::fmt::Debug for CustomConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "CustomConfig{{encode_table: {:?}, decode_table: {:?}, invalid_value: {:?}, padding_byte: {:?}}}", &self.encode_table[..], &self.decode_table[..], self.invalid_value, self.padding_byte)
    }
}

impl Padding for &CustomConfig {
    #[inline]
    fn padding_byte(self) -> Option<u8> {
        self.padding_byte
    }
}

impl ::private::Sealed for &CustomConfig {}