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
pub use encode::{encode, encode_config, encode_config_buf, encode_config_slice};

mod decode;
pub use decode::{decode, decode_config, decode_config_buf, decode_config_slice, DecodeError};

pub mod character_set;

#[cfg(test)]
mod tests;

pub const STANDARD:  Config<character_set::Standard, WithPadding> = Config{char_set: character_set::Standard, padding: WithPadding};
pub const STANDARD_NO_PAD: Config<character_set::Standard, NoPadding> = Config{char_set: character_set::Standard, padding: NoPadding};
pub const URL_SAFE: Config<character_set::UrlSafe, WithPadding> = Config{char_set: character_set::UrlSafe, padding: WithPadding};
pub const URL_SAFE_NO_PAD: Config<character_set::UrlSafe, NoPadding> = Config{char_set: character_set::UrlSafe, padding: NoPadding};
pub const CRYPT: Config<character_set::Crypt, WithPadding> = Config{char_set: character_set::Crypt, padding: WithPadding};
pub const CRYPT_NO_PAD: Config<character_set::Crypt, NoPadding> = Config{char_set: character_set::Crypt, padding: NoPadding};


pub trait Encoding : IntoBulkEncoding + Copy
where
    Self: Sized,
{
    fn encode_u6(self, input: u8) -> u8;
}

pub trait IntoBulkEncoding : Copy {
    type BulkEncoding: BulkEncoding;
    fn into_bulk_encoding(self) -> Self::BulkEncoding;
}

pub trait Decoding : Copy {
    const INVALID_VALUE: u8;
    fn decode_u8(self, input: u8) -> u8;
}

pub trait Padding : Copy {
    const PADDING_BYTE: u8 = b'=';
    fn has_padding(self) -> bool;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct WithPadding;
impl Padding for WithPadding {
    #[inline]
    fn has_padding(self) -> bool { true }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoPadding;
impl Padding for NoPadding {
    #[inline]
    fn has_padding(self) -> bool { false }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Config<C, P> where C: Copy, P: Copy {
    char_set: C,
    padding: P,
}

impl<C, P> Padding for Config<C, P> where P: Padding, C: Copy {
    const PADDING_BYTE: u8 = P::PADDING_BYTE;

    #[inline]
    fn has_padding(self) -> bool {
        self.padding.has_padding()
    }
}

impl<C, P> Encoding for Config<C, P> where C: Encoding, P: Copy {
    #[inline]
    fn encode_u6(self, input: u8) -> u8 {
        self.char_set.encode_u6(input)
    }
}

impl<C, P> IntoBulkEncoding for Config<C, P> where C: IntoBulkEncoding, P: Copy {
    type BulkEncoding = C::BulkEncoding;

    #[inline]
    fn into_bulk_encoding(self) -> Self::BulkEncoding {
        self.char_set.into_bulk_encoding()
    }
}

impl<C, P> Decoding for Config<C, P> where C: Decoding, P: Copy {
    const INVALID_VALUE: u8 = C::INVALID_VALUE;

    #[inline]
    fn decode_u8(self, input: u8) -> u8 {
        self.char_set.decode_u8(input)
    }
}

pub trait BulkEncoding {
    const MIN_INPUT_BYTES: usize;

    fn bulk_encode(self, input: &[u8], output: &mut [u8]) -> (usize, usize);
}