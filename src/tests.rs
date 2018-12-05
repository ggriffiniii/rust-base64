extern crate rand;

use encode::encoded_size;
use *;

use std::str;

use self::rand::distributions::{Distribution, Range};
use self::rand::{FromEntropy, Rng};

#[test]
fn roundtrip_random_config_short() {
    // exercise the slower encode/decode routines that operate on shorter buffers more vigorously
    roundtrip_random_config(Range::new(0, 50), 10_000);
}

#[test]
fn roundtrip_random_config_long() {
    roundtrip_random_config(Range::new(0, 1000), 10_000);
}

pub fn assert_encode_sanity<C: Encoding + Padding>(encoded: &str, config: C, input_len: usize) {
    let input_rem = input_len % 3;
    let expected_padding_len = if input_rem > 0 {
        if config.has_padding() {
            3 - input_rem
        } else {
            0
        }
    } else {
        0
    };

    let expected_encoded_len = encoded_size(input_len, config).unwrap();

    assert_eq!(expected_encoded_len, encoded.len());

    let padding_len = encoded.as_bytes().iter().filter(|&&c| c == C::PADDING_BYTE).count();

    assert_eq!(expected_padding_len, padding_len);

    let _ = str::from_utf8(encoded.as_bytes()).expect("Base64 should be valid utf8");
}

fn roundtrip_random_config(input_len_range: Range<usize>, iterations: u32) {
    let mut input_buf: Vec<u8> = Vec::new();
    let mut encoded_buf = String::new();
    let mut rng = rand::rngs::SmallRng::from_entropy();

    for _ in 0..iterations {
        input_buf.clear();
        encoded_buf.clear();

        let input_len = input_len_range.sample(&mut rng);

        let config = random_config(&mut rng);

        for _ in 0..input_len {
            input_buf.push(rng.gen());
        }

        encode_config_buf(&input_buf, config, &mut encoded_buf);

        assert_encode_sanity(&encoded_buf, config, input_len);

        assert_eq!(input_buf, decode_config(&encoded_buf, config).unwrap());
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Configs {
    Standard(Config<::character_set::Standard, ::WithPadding>),
    StandardNoPad(Config<::character_set::Standard, ::NoPadding>),
    UrlSafe(Config<::character_set::UrlSafe, ::WithPadding>),
    UrlSafeNoPad(Config<::character_set::UrlSafe, ::NoPadding>),
    Crypt(Config<::character_set::Crypt, ::WithPadding>),
    CryptNoPad(Config<::character_set::Crypt, ::NoPadding>),
}

impl Padding for Configs {
    const PADDING_BYTE: u8 = ::WithPadding::PADDING_BYTE;

    fn has_padding(self) -> bool {
        use self::Configs::*;
        match self {
            Standard(x) => x.has_padding(),
            StandardNoPad(x) => x.has_padding(),
            UrlSafe(x) => x.has_padding(),
            UrlSafeNoPad(x) => x.has_padding(),
            Crypt(x) => x.has_padding(),
            CryptNoPad(x) => x.has_padding(),
        }
    }
}

impl Encoding for Configs {
    fn encode_u6(self, input: u8) -> u8 {
        use self::Configs::*;
        match self {
            Standard(x) => x.encode_u6(input),
            StandardNoPad(x) => x.encode_u6(input),
            UrlSafe(x) => x.encode_u6(input),
            UrlSafeNoPad(x) => x.encode_u6(input),
            Crypt(x) => x.encode_u6(input),
            CryptNoPad(x) => x.encode_u6(input),
        }
    }
}

pub(crate) struct BulkEncoding(Configs);
impl ::BulkEncoding for BulkEncoding {
    const MIN_INPUT_BYTES: usize = 0;

    fn bulk_encode(self, input: &[u8], output: &mut [u8]) -> (usize, usize) {
        use self::Configs::*;
        match self.0 {
            Standard(x) => {
                let bulk_encoding = x.into_bulk_encoding();
                bulk_encoding.bulk_encode(input, output)
            },
            StandardNoPad(x) => {
                let bulk_encoding = x.into_bulk_encoding();
                bulk_encoding.bulk_encode(input, output)
            },
            UrlSafe(x) => {
                let bulk_encoding = x.into_bulk_encoding();
                bulk_encoding.bulk_encode(input, output)
            },
            UrlSafeNoPad(x) => {
                let bulk_encoding = x.into_bulk_encoding();
                bulk_encoding.bulk_encode(input, output)
            },
            Crypt(x) => {
                let bulk_encoding = x.into_bulk_encoding();
                bulk_encoding.bulk_encode(input, output)
            },
            CryptNoPad(x) => {
                let bulk_encoding = x.into_bulk_encoding();
                bulk_encoding.bulk_encode(input, output)
            },
        }
        
    }
}

impl IntoBulkEncoding for Configs {
    type BulkEncoding = BulkEncoding;

    fn into_bulk_encoding(self) -> Self::BulkEncoding {
        BulkEncoding(self)
    }
}

impl Decoding for Configs {
    const INVALID_VALUE: u8 = ::character_set::Standard::INVALID_VALUE;

    fn decode_u8(self, input: u8) -> u8 {
        use self::Configs::*;
        match self {
            Standard(x) => x.decode_u8(input),
            StandardNoPad(x) => x.decode_u8(input),
            UrlSafe(x) => x.decode_u8(input),
            UrlSafeNoPad(x) => x.decode_u8(input),
            Crypt(x) => x.decode_u8(input),
            CryptNoPad(x) => x.decode_u8(input),
        }
    }
}

impl rand::distributions::Distribution<Configs> for rand::distributions::Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Configs {
        use std::default::Default;
        use self::Configs::*;
        match rng.gen_range(0, 6) {
            0 => Standard(Default::default()),
            1 => StandardNoPad(Default::default()),
            2 => UrlSafe(Default::default()),
            3 => UrlSafeNoPad(Default::default()),
            4 => Crypt(Default::default()),
            5 => CryptNoPad(Default::default()),
            _ => unreachable!(),
        }
    }
}

pub(crate) fn random_config<R: Rng>(rng: &mut R) -> Configs {
    rng.gen()
}
