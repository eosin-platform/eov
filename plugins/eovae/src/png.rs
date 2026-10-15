//! A small PNG encoder for the batch heat maps: 8-bit RGBA, zlib from
//! `miniz_oxide`, and `tEXt` chunks for the map's scale and colour range.

/// Encodes `rgba` (`width·height·4` bytes, straight alpha) as a PNG with
/// the given text entries (keywords of 1–79 Latin-1 characters).
pub fn encode_rgba(width: u32, height: u32, rgba: &[u8], text: &[(&str, &str)]) -> Vec<u8> {
    assert!(
        width > 0 && height > 0,
        "PNG images have at least one pixel"
    );
    let stride = width as usize * 4;
    assert_eq!(rgba.len(), stride * height as usize, "RGBA size");
    let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    // 8 bits per channel, RGBA, deflate, adaptive filtering, no interlace.
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &header);
    for (keyword, value) in text {
        let mut data = Vec::with_capacity(keyword.len() + 1 + value.len());
        data.extend_from_slice(keyword.as_bytes());
        data.push(0);
        data.extend_from_slice(value.as_bytes());
        chunk(&mut png, b"tEXt", &data);
    }
    // Each row is filtered with "up" (rows of a heat map repeat a lot).
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    let mut previous: Option<&[u8]> = None;
    for row in rgba.chunks_exact(stride) {
        match previous {
            Some(above) => {
                raw.push(2);
                raw.extend(row.iter().zip(above).map(|(a, b)| a.wrapping_sub(*b)));
            }
            None => {
                raw.push(0);
                raw.extend_from_slice(row);
            }
        }
        previous = Some(row);
    }
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6);
    chunk(&mut png, b"IDAT", &compressed);
    chunk(&mut png, b"IEND", &[]);
    png
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    let crc = crc32(&[kind.as_slice(), data]);
    png.extend_from_slice(&crc.to_be_bytes());
}

/// CRC-32 (ISO 3309, as PNG uses) over consecutive byte slices.
fn crc32(parts: &[&[u8]]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (n, entry) in table.iter_mut().enumerate() {
            let mut c = n as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *entry = c;
        }
        table
    });
    let mut crc = 0xFFFF_FFFFu32;
    for part in parts {
        for byte in *part {
            crc = table[((crc ^ u32::from(*byte)) & 0xFF) as usize] ^ (crc >> 8);
        }
    }
    crc ^ 0xFFFF_FFFF
}

/// A PNG as [`decode_rgba`] reads it.
#[cfg(test)]
#[derive(Debug)]
pub struct Decoded {
    pub width: u32,
    pub height: u32,
    pub text: Vec<(String, String)>,
    pub rgba: Vec<u8>,
}

/// Reads what [`encode_rgba`] writes (for tests and checks).
#[cfg(test)]
pub fn decode_rgba(png: &[u8]) -> Result<Decoded, String> {
    if png.get(..8) != Some(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A][..]) {
        return Err("not a PNG".into());
    }
    let mut at = 8;
    let (mut width, mut height) = (0u32, 0u32);
    let mut text = Vec::new();
    let mut idat = Vec::new();
    while at + 12 <= png.len() {
        let len = u32::from_be_bytes(png[at..at + 4].try_into().unwrap()) as usize;
        let kind: [u8; 4] = png[at + 4..at + 8].try_into().unwrap();
        let data = png.get(at + 8..at + 8 + len).ok_or("truncated chunk")?;
        let crc = u32::from_be_bytes(
            png.get(at + 8 + len..at + 12 + len)
                .ok_or("truncated CRC")?
                .try_into()
                .unwrap(),
        );
        if crc != crc32(&[&kind, data]) {
            return Err(format!("bad CRC in {}", String::from_utf8_lossy(&kind)));
        }
        match &kind {
            b"IHDR" => {
                width = u32::from_be_bytes(data[0..4].try_into().unwrap());
                height = u32::from_be_bytes(data[4..8].try_into().unwrap());
                if data[8..13] != [8, 6, 0, 0, 0] {
                    return Err("not 8-bit RGBA".into());
                }
            }
            b"tEXt" => {
                let split = data.iter().position(|b| *b == 0).ok_or("bad tEXt")?;
                text.push((
                    String::from_utf8_lossy(&data[..split]).into_owned(),
                    String::from_utf8_lossy(&data[split + 1..]).into_owned(),
                ));
            }
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        at += 12 + len;
    }
    let raw = miniz_oxide::inflate::decompress_to_vec_zlib(&idat).map_err(|e| format!("{e:?}"))?;
    let stride = width as usize * 4;
    let mut rgba: Vec<u8> = Vec::with_capacity(stride * height as usize);
    for (y, row) in raw.chunks_exact(stride + 1).enumerate() {
        let (filter, bytes) = (row[0], &row[1..]);
        match filter {
            0 => rgba.extend_from_slice(bytes),
            2 => {
                let above = (y - 1) * stride;
                for (x, byte) in bytes.iter().enumerate() {
                    let up = rgba[above + x];
                    rgba.push(byte.wrapping_add(up));
                }
            }
            other => return Err(format!("unexpected filter {other}")),
        }
    }
    if rgba.len() != stride * height as usize {
        return Err("wrong pixel count".into());
    }
    Ok(Decoded {
        width,
        height,
        text,
        rgba,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_matches_the_png_specification() {
        // The CRC of an empty IEND chunk, as in every PNG.
        assert_eq!(crc32(&[b"IEND"]), 0xAE42_6082);
    }

    #[test]
    fn images_and_text_round_trip() {
        let (width, height) = (5, 3);
        let rgba: Vec<u8> = (0..width * height * 4)
            .map(|i| (i * 7 % 256) as u8)
            .collect();
        let png = encode_rgba(width, height, &rgba, &[("Software", "eovae"), ("k", "v=1")]);
        let decoded = decode_rgba(&png).unwrap();
        assert_eq!((decoded.width, decoded.height), (width, height));
        assert_eq!(decoded.rgba, rgba);
        assert_eq!(
            decoded.text[0],
            ("Software".to_string(), "eovae".to_string())
        );
        assert_eq!(decoded.text[1].1, "v=1");
    }
}
