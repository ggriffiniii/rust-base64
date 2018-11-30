use byteorder::{BigEndian, ByteOrder};
use {tables, CharacterSet, Config, STANDARD};

use std::{error, fmt, ops::Range, str};

// decode logic operates on chunks of 8 input bytes without padding
const INPUT_CHUNK_LEN: usize = 8;
const DECODED_CHUNK_LEN: usize = 6;
// we read a u64 and write a u64, but a u64 of input only yields 6 bytes of output, so the last
// 2 bytes of any output u64 should not be counted as written to (but must be available in a
// slice).
const DECODED_CHUNK_SUFFIX: usize = 2;

// how many u64's of input to handle at a time
const CHUNKS_PER_FAST_LOOP_BLOCK: usize = 4;
const INPUT_BLOCK_LEN: usize = CHUNKS_PER_FAST_LOOP_BLOCK * INPUT_CHUNK_LEN;
// includes the trailing 2 bytes for the final u64 write
const DECODED_BLOCK_LEN: usize =
    CHUNKS_PER_FAST_LOOP_BLOCK * DECODED_CHUNK_LEN + DECODED_CHUNK_SUFFIX;

/// Errors that can occur while decoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// An invalid byte was found in the input. The offset and offending byte are provided.
    InvalidByte(usize, u8),
    /// The length of the input is invalid.
    InvalidLength,
    /// The last non-padding input symbol's encoded 6 bits have nonzero bits that will be discarded.
    /// This is indicative of corrupted or truncated Base64.
    /// Unlike InvalidByte, which reports symbols that aren't in the alphabet, this error is for
    /// symbols that are in the alphabet but represent nonsensical encodings.
    InvalidLastSymbol(usize, u8),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            DecodeError::InvalidByte(index, byte) => {
                write!(f, "Invalid byte {}, offset {}.", byte, index)
            }
            DecodeError::InvalidLength => write!(f, "Encoded text cannot have a 6-bit remainder."),
            DecodeError::InvalidLastSymbol(index, byte) => {
                write!(f, "Invalid last symbol {}, offset {}.", byte, index)
            }
        }
    }
}

impl error::Error for DecodeError {
    fn description(&self) -> &str {
        match *self {
            DecodeError::InvalidByte(_, _) => "invalid byte",
            DecodeError::InvalidLength => "invalid length",
            DecodeError::InvalidLastSymbol(_, _) => "invalid last symbol",
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        None
    }
}

///Decode from string reference as octets.
///Returns a Result containing a Vec<u8>.
///Convenience `decode_config(input, base64::STANDARD);`.
///
///# Example
///
///```rust
///extern crate base64;
///
///fn main() {
///    let bytes = base64::decode("aGVsbG8gd29ybGQ=").unwrap();
///    println!("{:?}", bytes);
///}
///```
pub fn decode<T: ?Sized + AsRef<[u8]>>(input: &T) -> Result<Vec<u8>, DecodeError> {
    decode_config(input, STANDARD)
}

///Decode from string reference as octets.
///Returns a Result containing a Vec<u8>.
///
///# Example
///
///```rust
///extern crate base64;
///
///fn main() {
///    let bytes = base64::decode_config("aGVsbG8gd29ybGR+Cg==", base64::STANDARD).unwrap();
///    println!("{:?}", bytes);
///
///    let bytes_url = base64::decode_config("aGVsbG8gaW50ZXJuZXR-Cg==", base64::URL_SAFE).unwrap();
///    println!("{:?}", bytes_url);
///}
///```
pub fn decode_config<T: ?Sized + AsRef<[u8]>>(
    input: &T,
    config: Config,
) -> Result<Vec<u8>, DecodeError> {
    let mut buffer = Vec::<u8>::new();

    decode_config_buf(input, config, &mut buffer).map(|_| buffer)
}

///Decode from string reference as octets.
///Writes into the supplied buffer to avoid allocation.
///Returns a Result containing an empty tuple, aka ().
///
///# Example
///
///```rust
///extern crate base64;
///
///fn main() {
///    let mut buffer = Vec::<u8>::new();
///    base64::decode_config_buf("aGVsbG8gd29ybGR+Cg==", base64::STANDARD, &mut buffer).unwrap();
///    println!("{:?}", buffer);
///
///    buffer.clear();
///
///    base64::decode_config_buf("aGVsbG8gaW50ZXJuZXR-Cg==", base64::URL_SAFE, &mut buffer)
///        .unwrap();
///    println!("{:?}", buffer);
///}
///```
pub fn decode_config_buf<T: ?Sized + AsRef<[u8]>>(
    input: &T,
    config: Config,
    buffer: &mut Vec<u8>,
) -> Result<(), DecodeError> {
    let input_bytes = input.as_ref();

    let starting_output_len = buffer.len();

    let num_chunks = num_chunks(input_bytes);
    let decoded_len_estimate = num_chunks
        .checked_mul(DECODED_CHUNK_LEN)
        .and_then(|p| p.checked_add(starting_output_len))
        .expect("Overflow when calculating output buffer length");
    buffer.resize(decoded_len_estimate, 0);

    let bytes_written;
    {
        let buffer_slice = &mut buffer.as_mut_slice()[starting_output_len..];
        bytes_written = decode_config_slice(input_bytes, config, buffer_slice)?;
    }

    buffer.truncate(starting_output_len + bytes_written);

    Ok(())
}

/// Return the number of input chunks (including a possibly partial final chunk) in the input
fn num_chunks(input: &[u8]) -> usize {
    input
        .len()
        .checked_add(INPUT_CHUNK_LEN - 1)
        .expect("Overflow when calculating number of chunks in input")
        / INPUT_CHUNK_LEN
}

trait UsizeRangeExt {
    fn advance(self, step: usize) -> Range<usize>;
    fn is_within(&self, &[u8]) -> bool;
}

impl UsizeRangeExt for Range<usize> {
    #[inline]
    fn advance(self, step: usize) -> Range<usize> {
        self.start + step..self.end + step
    }

    #[inline]
    fn is_within(&self, s: &[u8]) -> bool {
        self.end <= s.len()
    }
}

/// Decode the input into the provided output slice.
///
/// This will not write any bytes past exactly what is decoded (no stray garbage bytes at the end).
///
/// If you don't know ahead of time what the decoded length should be, size your buffer with a
/// conservative estimate for the decoded length of an input: 3 bytes of output for every 4 bytes of
/// input, rounded up, or in other words `(input_len + 3) / 4 * 3`.
///
/// If the slice is not large enough, this will panic.
pub fn decode_config_slice<T: ?Sized + AsRef<[u8]>>(
    input: &T,
    config: Config,
    output: &mut [u8],
) -> Result<usize, DecodeError> {
    let input: &[u8] = input.as_ref();
    let decode_table = config.char_set.decode_table();
    let (input_index, mut output_index) = {
        // The fast decode loop does not handle padding. If the input is a multiple
        // of INPUT_CHUNK_LEN then don't process the last chunk, otherwise ignore
        // the trailing partial chunk.
        let remainder_len = input.len() % INPUT_CHUNK_LEN;
        let trailing_bytes_to_skip = match remainder_len {
            // if input is a multiple of the chunk size, ignore the last chunk as it may have padding,
            // and the fast decode logic cannot handle padding
            0 => INPUT_CHUNK_LEN,
            // 1 and 5 trailing bytes are illegal: can't decode 6 bits of input into a byte
            1 | 5 => return Err(DecodeError::InvalidLength),
            // Ignore the trailing partial chunk.
            2 | 3 | 4 | 6 | 7 => remainder_len,
            _ => unreachable!(),
        };
        let fast_loop_input = &input[..input.len().saturating_sub(trailing_bytes_to_skip)];
        // Fast loop, stage 1.
        // bulk_decode accepts a input slice that's a multiple of INPUT_CHUNK_LEN
        // and processes it in chunks that are a multiple of INPUT_CHUNK_LEN. The
        // corrolary is that the remaining bytes of fast_loop_input after
        // bulk_decode is also a multiple of INPUT_CHUNK_LEN.
        let (mut input_index, mut output_index) =
            bulk_decode(fast_loop_input, config.char_set, output)?;

        // Fast loop, stage 2 (aka still pretty fast loop)
        // 8 bytes at a time for whatever we didn't do in stage 1.
        let mut input_range = input_index..input_index + INPUT_CHUNK_LEN;
        let mut output_range =
            output_index..output_index + DECODED_CHUNK_LEN + DECODED_CHUNK_SUFFIX;
        while input_range.is_within(fast_loop_input) && output_range.is_within(output) {
            decode_chunk(
                &fast_loop_input[input_range.clone()],
                input_range.start,
                decode_table,
                &mut output[output_range.clone()],
            )?;

            input_range = input_range.advance(INPUT_CHUNK_LEN);
            output_range = output_range.advance(DECODED_CHUNK_LEN);
        }

        // Stage 3
        // It's possible when processing stage 2 that there are 8 bytes remaining in
        // the input slice, but less than 8 bytes remaining in the output slice.
        // Since stage 2 outputs 8 bytes (only 6 of which are valid) for every 8
        // bytes of input, it can stop short of processing the entire
        // fast_loop_input because the output buffer is not large enough. In this
        // case use decode_chunk_precise which outputs precisely 6 bytes.
        output_range = output_range.start..output_range.start + DECODED_CHUNK_LEN;
        if input_range.is_within(fast_loop_input) {
            decode_chunk_precise(
                &input[input_range.clone()],
                input_range.start,
                decode_table,
                &mut output[output_range.clone()],
            )?;
            input_range = input_range.advance(INPUT_CHUNK_LEN);
            output_range = output_range.advance(DECODED_CHUNK_LEN);
        }

        // fast_loop_input has been fully processed. All that remains is the last (possibly partial) chunk.
        debug_assert_eq!(fast_loop_input.len() - input_range.start, 0);
        debug_assert!(
            input.len() - input_range.start == trailing_bytes_to_skip || input.is_empty()
        );
        debug_assert!(input.len() - input_range.start > 1 || input.is_empty());
        debug_assert!(input.len() - input_range.start <= 8);
        (input_range.start, output_range.start)
    };

    // Stage 4
    // Finally, decode any leftovers that aren't a complete input block of 8 bytes.
    // Use a u64 as a stack-resident 8 byte buffer.
    let mut leftover_bits: u64 = 0;
    let mut morsels_in_leftover = 0;
    let mut padding_bytes = 0;
    let mut first_padding_index: usize = 0;
    let mut last_symbol = 0_u8;
    let start_of_leftovers = input_index;
    for (i, b) in input[start_of_leftovers..].iter().enumerate() {
        // '=' padding
        if *b == 0x3D {
            // There can be bad padding in a few ways:
            // 1 - Padding with non-padding characters after it
            // 2 - Padding after zero or one non-padding characters before it
            //     in the current quad.
            // 3 - More than two characters of padding. If 3 or 4 padding chars
            //     are in the same quad, that implies it will be caught by #2.
            //     If it spreads from one quad to another, it will be caught by
            //     #2 in the second quad.

            if i % 4 < 2 {
                // Check for case #2.
                let bad_padding_index = start_of_leftovers + if padding_bytes > 0 {
                    // If we've already seen padding, report the first padding index.
                    // This is to be consistent with the faster logic above: it will report an
                    // error on the first padding character (since it doesn't expect to see
                    // anything but actual encoded data).
                    first_padding_index
                } else {
                    // haven't seen padding before, just use where we are now
                    i
                };
                return Err(DecodeError::InvalidByte(bad_padding_index, *b));
            }

            if padding_bytes == 0 {
                first_padding_index = i;
            }

            padding_bytes += 1;
            continue;
        }

        // Check for case #1.
        // To make '=' handling consistent with the main loop, don't allow
        // non-suffix '=' in trailing chunk either. Report error as first
        // erroneous padding.
        if padding_bytes > 0 {
            return Err(DecodeError::InvalidByte(
                start_of_leftovers + first_padding_index,
                0x3D,
            ));
        }
        last_symbol = *b;

        // can use up to 8 * 6 = 48 bits of the u64, if last chunk has no padding.
        // To minimize shifts, pack the leftovers from left to right.
        let shift = 64 - (morsels_in_leftover + 1) * 6;
        // tables are all 256 elements, lookup with a u8 index always succeeds
        let morsel = decode_table[*b as usize];
        if morsel == tables::INVALID_VALUE {
            return Err(DecodeError::InvalidByte(start_of_leftovers + i, *b));
        }

        leftover_bits |= (morsel as u64) << shift;
        morsels_in_leftover += 1;
    }

    let leftover_bits_ready_to_append = match morsels_in_leftover {
        0 => 0,
        2 => 8,
        3 => 16,
        4 => 24,
        6 => 32,
        7 => 40,
        8 => 48,
        _ => unreachable!(
            "Impossible: must only have 0 to 8 input bytes in last chunk, with no invalid lengths"
        ),
    };

    // if there are bits set outside the bits we care about, last symbol encodes trailing bits that
    // will not be included in the output
    let mask = !0 >> leftover_bits_ready_to_append;
    if (leftover_bits & mask) != 0 {
        // last morsel is at `morsels_in_leftover` - 1
        return Err(DecodeError::InvalidLastSymbol(
            start_of_leftovers + morsels_in_leftover - 1,
            last_symbol,
        ));
    }

    let mut leftover_bits_appended_to_buf = 0;
    while leftover_bits_appended_to_buf < leftover_bits_ready_to_append {
        // `as` simply truncates the higher bits, which is what we want here
        let selected_bits = (leftover_bits >> (56 - leftover_bits_appended_to_buf)) as u8;
        output[output_index] = selected_bits;
        output_index += 1;

        leftover_bits_appended_to_buf += 8;
    }

    Ok(output_index)
}

#[inline(always)]
fn bulk_decode(
    input: &[u8],
    char_set: CharacterSet,
    output: &mut [u8],
) -> Result<(usize, usize), DecodeError> {
    #[cfg(all(
        feature = "simd",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    let (input_index, output_index) = {
        if input.len() >= avx2::REGISTER_BYTES && is_x86_feature_detected!("avx2") {
            unsafe { avx2::bulk_decode(input, char_set, output) }
        } else {
            (0, 0)
        }
    };
    let decode_table = char_set.decode_table();

    let mut input_range = input_index..input_index+INPUT_BLOCK_LEN;
    let mut output_range = output_index..output_index+DECODED_BLOCK_LEN;
    while input_range.is_within(input) && output_range.is_within(output) {
        let input_slice = &input[input_range.clone()];
        let output_slice = &mut output[output_range.clone()];

        decode_chunk(
            &input_slice[0..],
            input_range.start,
            decode_table,
            &mut output_slice[0..],
        )?;
        decode_chunk(
            &input_slice[8..],
            input_range.start + 8,
            decode_table,
            &mut output_slice[6..],
        )?;
        decode_chunk(
            &input_slice[16..],
            input_range.start + 16,
            decode_table,
            &mut output_slice[12..],
        )?;
        decode_chunk(
            &input_slice[24..],
            input_range.start + 24,
            decode_table,
            &mut output_slice[18..],
        )?;

        input_range = input_range.advance(INPUT_BLOCK_LEN);
        output_range = output_range.advance(DECODED_BLOCK_LEN - DECODED_CHUNK_SUFFIX);
    }
    Ok((input_range.start, output_range.start))
}

/// Decode 8 bytes of input into 6 bytes of output. 8 bytes of output will be written, but only the
/// first 6 of those contain meaningful data.
///
/// `input` is the bytes to decode, of which the first 8 bytes will be processed.
/// `index_at_start_of_input` is the offset in the overall input (used for reporting errors
/// accurately)
/// `decode_table` is the lookup table for the particular base64 alphabet.
/// `output` will have its first 8 bytes overwritten, of which only the first 6 are valid decoded
/// data.
// yes, really inline (worth 30-50% speedup)
#[inline(always)]
fn decode_chunk(
    input: &[u8],
    index_at_start_of_input: usize,
    decode_table: &[u8; 256],
    output: &mut [u8],
) -> Result<(), DecodeError> {
    let mut accum: u64;

    let morsel = decode_table[input[0] as usize];
    if morsel == tables::INVALID_VALUE {
        return Err(DecodeError::InvalidByte(index_at_start_of_input, input[0]));
    }
    accum = (morsel as u64) << 58;

    let morsel = decode_table[input[1] as usize];
    if morsel == tables::INVALID_VALUE {
        return Err(DecodeError::InvalidByte(
            index_at_start_of_input + 1,
            input[1],
        ));
    }
    accum |= (morsel as u64) << 52;

    let morsel = decode_table[input[2] as usize];
    if morsel == tables::INVALID_VALUE {
        return Err(DecodeError::InvalidByte(
            index_at_start_of_input + 2,
            input[2],
        ));
    }
    accum |= (morsel as u64) << 46;

    let morsel = decode_table[input[3] as usize];
    if morsel == tables::INVALID_VALUE {
        return Err(DecodeError::InvalidByte(
            index_at_start_of_input + 3,
            input[3],
        ));
    }
    accum |= (morsel as u64) << 40;

    let morsel = decode_table[input[4] as usize];
    if morsel == tables::INVALID_VALUE {
        return Err(DecodeError::InvalidByte(
            index_at_start_of_input + 4,
            input[4],
        ));
    }
    accum |= (morsel as u64) << 34;

    let morsel = decode_table[input[5] as usize];
    if morsel == tables::INVALID_VALUE {
        return Err(DecodeError::InvalidByte(
            index_at_start_of_input + 5,
            input[5],
        ));
    }
    accum |= (morsel as u64) << 28;

    let morsel = decode_table[input[6] as usize];
    if morsel == tables::INVALID_VALUE {
        return Err(DecodeError::InvalidByte(
            index_at_start_of_input + 6,
            input[6],
        ));
    }
    accum |= (morsel as u64) << 22;

    let morsel = decode_table[input[7] as usize];
    if morsel == tables::INVALID_VALUE {
        return Err(DecodeError::InvalidByte(
            index_at_start_of_input + 7,
            input[7],
        ));
    }
    accum |= (morsel as u64) << 16;

    BigEndian::write_u64(output, accum);

    Ok(())
}

/// Decode an 8-byte chunk, but only write the 6 bytes actually decoded instead of including 2
/// trailing garbage bytes.
#[inline]
fn decode_chunk_precise(
    input: &[u8],
    index_at_start_of_input: usize,
    decode_table: &[u8; 256],
    output: &mut [u8],
) -> Result<(), DecodeError> {
    let mut tmp_buf = [0_u8; 8];

    decode_chunk(
        input,
        index_at_start_of_input,
        decode_table,
        &mut tmp_buf[..],
    )?;

    output[0..6].copy_from_slice(&tmp_buf[0..6]);

    Ok(())
}

#[cfg(all(
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod avx2 {
    pub(super) const REGISTER_BYTES: usize = 32;

    use super::{CharacterSet, UsizeRangeExt};
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn bulk_decode(
        input: &[u8],
        char_set: CharacterSet,
        output: &mut [u8],
    ) -> (usize, usize) {
        let mut input_range = 0..REGISTER_BYTES;
        let mut output_range = 0..REGISTER_BYTES;
        while input_range.is_within(input) && output_range.is_within(output) {
            #[cfg_attr(feature = "cargo-clippy", allow(cast_ptr_alignment))]
            let mut data = _mm256_loadu_si256(input.as_ptr().add(input_range.start) as *const __m256i);

            let translate_result = match char_set {
                CharacterSet::Standard => translate_mm256i_standard(data),
                CharacterSet::UrlSafe => translate_mm256i_urlsafe(data),
                CharacterSet::Crypt => translate_mm256i_crypt(data),
            };
            data = match translate_result {
                Ok(data) => data,
                Err(_) => {
                    println!("error");
                    return (input_range.start, output_range.start)
                },
            };

            data = _mm256_maddubs_epi16(data, _mm256_set1_epi32(0x01400140));
            data = _mm256_madd_epi16(data, _mm256_set1_epi32(0x00011000));
            data = _mm256_shuffle_epi8(
                data,
                _mm256_setr_epi8(
                    2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1,
                    2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1),
            );
            data = _mm256_permutevar8x32_epi32(data, _mm256_setr_epi32(0, 1, 2, 4, 5, 6, -1, -1));
            #[cfg_attr(feature = "cargo-clippy", allow(cast_ptr_alignment))]
            _mm256_storeu_si256(output.as_mut_ptr().add(output_range.start) as *mut __m256i, data);
            input_range = input_range.advance(REGISTER_BYTES);
            output_range = output_range.advance(24);
        }
        (input_range.start, output_range.start)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(overflowing_literals)]
    unsafe fn translate_mm256i_standard(input: __m256i) -> Result<__m256i, ()> {
        let hi_nibbles = _mm256_and_si256(_mm256_srli_epi32(input, 4), _mm256_set1_epi8(0x0f));
        let low_nibbles = _mm256_and_si256(input, _mm256_set1_epi8(0x0f));
        let shift_lut = _mm256_setr_epi8(
            0,   0,  19,   4, -65, -65, -71, -71,
            0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,  19,   4, -65, -65, -71, -71,
            0,   0,   0,   0,   0,   0,   0,   0
        );
        
        let mask_lut = _mm256_setr_epi8(
        /* 0        */ 0b10101000,
        /* 1 .. 9   */ 0b11111000, 0b11111000, 0b11111000, 0b11111000,
                       0b11111000, 0b11111000, 0b11111000, 0b11111000,
                       0b11111000,
        /* 10       */ 0b11110000,
        /* 11       */ 0b01010100,
        /* 12 .. 14 */ 0b01010000, 0b01010000, 0b01010000,
        /* 15       */ 0b01010100,

        /* 0        */ 0b10101000,
        /* 1 .. 9   */ 0b11111000, 0b11111000, 0b11111000, 0b11111000,
                       0b11111000, 0b11111000, 0b11111000, 0b11111000,
                       0b11111000,
        /* 10       */ 0b11110000,
        /* 11       */ 0b01010100,
        /* 12 .. 14 */ 0b01010000, 0b01010000, 0b01010000,
        /* 15       */ 0b01010100
        );

        let bit_pos_lut = _mm256_setr_epi8(
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        );

        let sh = _mm256_shuffle_epi8(shift_lut,  hi_nibbles);
        let eq_slash  = _mm256_cmpeq_epi8(input, _mm256_set1_epi8(b'/' as i8));
        let shift  = _mm256_blendv_epi8(sh, _mm256_set1_epi8(16), eq_slash);
        let m      = _mm256_shuffle_epi8(mask_lut, low_nibbles);
        let bit    = _mm256_shuffle_epi8(bit_pos_lut, hi_nibbles);
        let non_match = _mm256_cmpeq_epi8(_mm256_and_si256(m, bit), _mm256_setzero_si256());
        if _mm256_movemask_epi8(non_match) != 0 {
            return Err(());
        }
        Ok(_mm256_add_epi8(input, shift))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(overflowing_literals)]
    unsafe fn translate_mm256i_urlsafe(input: __m256i) -> Result<__m256i, ()> {
        let hi_nibbles = _mm256_and_si256(_mm256_srli_epi32(input, 4), _mm256_set1_epi8(0x0f));
        let low_nibbles = _mm256_and_si256(input, _mm256_set1_epi8(0x0f));
        let shift_lut = _mm256_setr_epi8(
            0,   0,  17,   4, -65, -65, -71, -71,
            0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,  17,   4, -65, -65, -71, -71,
            0,   0,   0,   0,   0,   0,   0,   0
        );
        
        let mask_lut = _mm256_setr_epi8(
        /* 0        */ 0b10101000,
        /* 1 .. 9   */ 0b11111000, 0b11111000, 0b11111000, 0b11111000,
                       0b11111000, 0b11111000, 0b11111000, 0b11111000,
                       0b11111000,
        /* 10       */ 0b11110000,
        /* 11 .. 12 */ 0b01010000, 0b01010000,
        /* 13       */ 0b01010100,
        /* 14       */ 0b01010000,
        /* 15       */ 0b01110000,

        /* 0        */ 0b10101000,
        /* 1 .. 9   */ 0b11111000, 0b11111000, 0b11111000, 0b11111000,
                       0b11111000, 0b11111000, 0b11111000, 0b11111000,
                       0b11111000,
        /* 10       */ 0b11110000,
        /* 11 .. 12 */ 0b01010000, 0b01010000,
        /* 13       */ 0b01010100,
        /* 14       */ 0b01010000,
        /* 15       */ 0b01110000
        );

        let bit_pos_lut = _mm256_setr_epi8(
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        );

        let sh = _mm256_shuffle_epi8(shift_lut,  hi_nibbles);
        let eq_underscore  = _mm256_cmpeq_epi8(input, _mm256_set1_epi8(b'_' as i8));
        let shift  = _mm256_blendv_epi8(sh, _mm256_set1_epi8(-32), eq_underscore);
        let m      = _mm256_shuffle_epi8(mask_lut, low_nibbles);
        let bit    = _mm256_shuffle_epi8(bit_pos_lut, hi_nibbles);
        let non_match = _mm256_cmpeq_epi8(_mm256_and_si256(m, bit), _mm256_setzero_si256());
        if _mm256_movemask_epi8(non_match) != 0 {
            return Err(());
        }
        Ok(_mm256_add_epi8(input, shift))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(overflowing_literals)]
    unsafe fn translate_mm256i_crypt(input: __m256i) -> Result<__m256i, ()> {
        let hi_nibbles = _mm256_and_si256(_mm256_srli_epi32(input, 4), _mm256_set1_epi8(0x0f));
        let low_nibbles = _mm256_and_si256(input, _mm256_set1_epi8(0x0f));
        let shift_lut = _mm256_setr_epi8(
            0,   0,  -46,   -46, -53, -53, -69, -69,
            0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,  -46,   -46, -53, -53, -69, -69,
            0,   0,   0,   0,   0,   0,   0,   0
        );
        
        let mask_lut = _mm256_setr_epi8(
        /* 0        */ 0b10101000,
        /* 1 .. 9   */ 0b11111000, 0b11111000, 0b11111000, 0b11111000,
                       0b11111000, 0b11111000, 0b11111000, 0b11111000,
                       0b11111000,
        /* 10       */ 0b11110000,
        /* 11 .. 13 */ 0b01010000, 0b01010000, 0b01010000,
        /* 14 .. 15 */ 0b01010100, 0b01010100,

        /* 0        */ 0b10101000,
        /* 1 .. 9   */ 0b11111000, 0b11111000, 0b11111000, 0b11111000,
                       0b11111000, 0b11111000, 0b11111000, 0b11111000,
                       0b11111000,
        /* 10       */ 0b11110000,
        /* 11 .. 13 */ 0b01010000, 0b01010000, 0b01010000,
        /* 14 .. 15 */ 0b01010100, 0b01010100
        );

        let bit_pos_lut = _mm256_setr_epi8(
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        );

        let sh = _mm256_shuffle_epi8(shift_lut,  hi_nibbles);
        let m      = _mm256_shuffle_epi8(mask_lut, low_nibbles);
        let bit    = _mm256_shuffle_epi8(bit_pos_lut, hi_nibbles);
        let non_match = _mm256_cmpeq_epi8(_mm256_and_si256(m, bit), _mm256_setzero_si256());
        if _mm256_movemask_epi8(non_match) != 0 {
            return Err(());
        }
        Ok(_mm256_add_epi8(input, sh))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_chunk_precise_writes_only_6_bytes() {
        let input = b"Zm9vYmFy"; // "foobar"
        let mut output = [0_u8, 1, 2, 3, 4, 5, 6, 7];
        decode_chunk_precise(&input[..], 0, tables::STANDARD_DECODE, &mut output).unwrap();
        assert_eq!(&vec![b'f', b'o', b'o', b'b', b'a', b'r', 6, 7], &output);
    }

    #[test]
    fn decode_chunk_writes_8_bytes() {
        let input = b"Zm9vYmFy"; // "foobar"
        let mut output = [0_u8, 1, 2, 3, 4, 5, 6, 7];
        decode_chunk(&input[..], 0, tables::STANDARD_DECODE, &mut output).unwrap();
        assert_eq!(&vec![b'f', b'o', b'o', b'b', b'a', b'r', 0, 0], &output);
    }

    #[test]
    fn detect_invalid_last_symbol_two_bytes() {
        // example from https://github.com/alicemaz/rust-base64/issues/75
        assert!(decode("iYU=").is_ok());
        // trailing 01
        assert_eq!(Err(DecodeError::InvalidLastSymbol(2, b'V')), decode("iYV="));
        // trailing 10
        assert_eq!(Err(DecodeError::InvalidLastSymbol(2, b'W')), decode("iYW="));
        // trailing 11
        assert_eq!(Err(DecodeError::InvalidLastSymbol(2, b'X')), decode("iYX="));

        // also works when there are 2 quads in the last block
        assert_eq!(
            Err(DecodeError::InvalidLastSymbol(6, b'X')),
            decode("AAAAiYX=")
        );
    }

    #[test]
    fn detect_invalid_last_symbol_one_byte() {
        // 0xFF -> "/w==", so all letters > w, 0-9, and '+', '/' should get InvalidLastSymbol

        assert!(decode("/w==").is_ok());
        // trailing 01
        assert_eq!(Err(DecodeError::InvalidLastSymbol(1, b'x')), decode("/x=="));
        assert_eq!(Err(DecodeError::InvalidLastSymbol(1, b'z')), decode("/z=="));
        assert_eq!(Err(DecodeError::InvalidLastSymbol(1, b'0')), decode("/0=="));
        assert_eq!(Err(DecodeError::InvalidLastSymbol(1, b'9')), decode("/9=="));
        assert_eq!(Err(DecodeError::InvalidLastSymbol(1, b'+')), decode("/+=="));
        assert_eq!(Err(DecodeError::InvalidLastSymbol(1, b'/')), decode("//=="));

        // also works when there are 2 quads in the last block
        assert_eq!(
            Err(DecodeError::InvalidLastSymbol(5, b'x')),
            decode("AAAA/x==")
        );
    }

    #[test]
    fn detect_invalid_last_symbol_every_possible_three_symbols() {
        let mut base64_to_bytes = ::std::collections::HashMap::new();

        let mut bytes = [0_u8; 2];
        for b1 in 0_u16..256 {
            bytes[0] = b1 as u8;
            for b2 in 0_u16..256 {
                bytes[1] = b2 as u8;
                let mut b64 = vec![0_u8; 4];
                assert_eq!(4, ::encode_config_slice(&bytes, STANDARD, &mut b64[..]));
                let mut v = ::std::vec::Vec::with_capacity(2);
                v.extend_from_slice(&bytes[..]);

                assert!(base64_to_bytes.insert(b64, v).is_none());
            }
        }

        // every possible combination of symbols must either decode to 2 bytes or get InvalidLastSymbol

        let mut symbols = [0_u8; 4];
        for &s1 in STANDARD.char_set.encode_table().iter() {
            symbols[0] = s1;
            for &s2 in STANDARD.char_set.encode_table().iter() {
                symbols[1] = s2;
                for &s3 in STANDARD.char_set.encode_table().iter() {
                    symbols[2] = s3;
                    symbols[3] = b'=';

                    match base64_to_bytes.get(&symbols[..]) {
                        Some(bytes) => {
                            assert_eq!(Ok(bytes.to_vec()), decode_config(&symbols, STANDARD))
                        }
                        None => assert_eq!(
                            Err(DecodeError::InvalidLastSymbol(2, s3)),
                            decode_config(&symbols[..], STANDARD)
                        ),
                    }
                }
            }
        }
    }

    #[test]
    fn detect_invalid_last_symbol_every_possible_two_symbols() {
        let mut base64_to_bytes = ::std::collections::HashMap::new();

        for b in 0_u16..256 {
            let mut b64 = vec![0_u8; 4];
            assert_eq!(4, ::encode_config_slice(&[b as u8], STANDARD, &mut b64[..]));
            let mut v = ::std::vec::Vec::with_capacity(1);
            v.push(b as u8);

            assert!(base64_to_bytes.insert(b64, v).is_none());
        }

        // every possible combination of symbols must either decode to 1 byte or get InvalidLastSymbol

        let mut symbols = [0_u8; 4];
        for &s1 in STANDARD.char_set.encode_table().iter() {
            symbols[0] = s1;
            for &s2 in STANDARD.char_set.encode_table().iter() {
                symbols[1] = s2;
                symbols[2] = b'=';
                symbols[3] = b'=';

                match base64_to_bytes.get(&symbols[..]) {
                    Some(bytes) => {
                        assert_eq!(Ok(bytes.to_vec()), decode_config(&symbols, STANDARD))
                    }
                    None => assert_eq!(
                        Err(DecodeError::InvalidLastSymbol(1, s2)),
                        decode_config(&symbols[..], STANDARD)
                    ),
                }
            }
        }
    }
}
