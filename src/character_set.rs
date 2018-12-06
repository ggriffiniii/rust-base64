//! character_set
use {tables, Encoding, Decoding};

#[derive(Debug, Default, Clone, Copy)]
pub struct Standard;

#[derive(Debug, Default, Clone, Copy)]
pub struct UrlSafe;

#[derive(Debug, Default, Clone, Copy)]
pub struct Crypt;

#[inline]
fn encode_u6_by_table(input: u8, encode_table: &[u8;64]) -> u8 {
    debug_assert!(input < 64);
    encode_table[input as usize]
}

impl Encoding for Standard {
    #[inline]
    fn encode_u6(self, input: u8) -> u8 {
        encode_u6_by_table(input, tables::STANDARD_ENCODE)
    }
}

impl Encoding for UrlSafe {
    #[inline]
    fn encode_u6(self, input: u8) -> u8 {
        encode_u6_by_table(input, tables::URL_SAFE_ENCODE)
    }
}

impl Encoding for Crypt {
    #[inline]
    fn encode_u6(self, input: u8) -> u8 {
        encode_u6_by_table(input, tables::CRYPT_ENCODE)
    }
}

#[inline]
fn decode_by_table(input: u8, decode_table: &[u8;256]) -> u8 {
    decode_table[input as usize]
}

impl Decoding for Standard {
    #[inline]
    fn decode_u8(self, input: u8) -> u8 {
        decode_by_table(input, tables::STANDARD_DECODE)
    }

    #[inline]
    fn invalid_value(self) -> u8 {
        tables::INVALID_VALUE
    }
}

impl Decoding for UrlSafe {
    #[inline]
    fn decode_u8(self, input: u8) -> u8 {
        decode_by_table(input, tables::URL_SAFE_DECODE)
    }

    #[inline]
    fn invalid_value(self) -> u8 {
        tables::INVALID_VALUE
    }
}

impl Decoding for Crypt {
    #[inline]
    fn decode_u8(self, input: u8) -> u8 {
        decode_by_table(input, tables::CRYPT_DECODE)
    }

    #[inline]
    fn invalid_value(self) -> u8 {
        tables::INVALID_VALUE
    }
}