//! A small reader for the parts of an ONNX model eovae needs: graph inputs
//! and outputs with types and shapes, `metadata_props`, opsets and the
//! external-data files holding the weights.
//!
//! It decodes the protobuf wire format directly and skips large fields
//! (weights embedded in `raw_data`) by seeking, so inspecting a model never
//! needs ONNX Runtime, `protoc` or the whole file in memory.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Component, Path};

/// ONNX `TensorProto.DataType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElemType {
    Float,
    Float16,
    BFloat16,
    Double,
    Other(i32),
}

impl ElemType {
    pub fn from_onnx(value: i32) -> Self {
        match value {
            1 => Self::Float,
            10 => Self::Float16,
            11 => Self::Double,
            16 => Self::BFloat16,
            other => Self::Other(other),
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::Float => "float32".to_string(),
            Self::Float16 => "float16".to_string(),
            Self::BFloat16 => "bfloat16".to_string(),
            Self::Double => "float64".to_string(),
            Self::Other(0) => "undefined".to_string(),
            Self::Other(code) => {
                let name = match code {
                    2 => "uint8",
                    3 => "int8",
                    4 => "uint16",
                    5 => "int16",
                    6 => "int32",
                    7 => "int64",
                    8 => "string",
                    9 => "bool",
                    12 => "uint32",
                    13 => "uint64",
                    _ => return format!("type {code}"),
                };
                name.to_string()
            }
        }
    }
}

/// One tensor dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    Fixed(i64),
    Symbolic(String),
    Unknown,
}

impl Dim {
    pub fn fixed(&self) -> Option<i64> {
        match self {
            Dim::Fixed(value) if *value > 0 => Some(*value),
            _ => None,
        }
    }

    pub fn label(&self) -> String {
        match self {
            Dim::Fixed(value) => value.to_string(),
            Dim::Symbolic(name) if !name.is_empty() => name.clone(),
            _ => "?".to_string(),
        }
    }
}

/// A graph input or output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// `None` for non-tensor values (sequences, maps).
    pub elem_type: Option<ElemType>,
    /// `None` when the shape is not declared.
    pub shape: Option<Vec<Dim>>,
}

impl TensorInfo {
    pub fn shape_label(&self) -> String {
        match &self.shape {
            Some(dims) => format!(
                "[{}]",
                dims.iter().map(Dim::label).collect::<Vec<_>>().join(" × ")
            ),
            None => "[?]".to_string(),
        }
    }

    pub fn type_label(&self) -> String {
        self.elem_type
            .map(ElemType::label)
            .unwrap_or_else(|| "non-tensor".to_string())
    }
}

/// What eovae knows about a model file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OnnxSummary {
    pub ir_version: i64,
    pub producer: String,
    pub opsets: Vec<(String, i64)>,
    pub metadata: BTreeMap<String, String>,
    /// Graph inputs that are not initializers (what must be fed).
    pub inputs: Vec<TensorInfo>,
    pub outputs: Vec<TensorInfo>,
    /// External-data file locations, relative to the model, in first-use
    /// order.
    pub external_data: Vec<String>,
}

const MAX_STRING: u64 = 1 << 20;

/// A protobuf wire-format reader over a seekable source.
struct Wire<R> {
    inner: R,
    pos: u64,
}

impl<R: Read + Seek> Wire<R> {
    fn new(inner: R) -> Self {
        Self { inner, pos: 0 }
    }

    fn byte(&mut self) -> io::Result<u8> {
        let mut buffer = [0u8; 1];
        self.inner.read_exact(&mut buffer)?;
        self.pos += 1;
        Ok(buffer[0])
    }

    fn varint(&mut self) -> io::Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(invalid("varint is too long"))
    }

    fn skip(&mut self, count: u64) -> io::Result<()> {
        let count = i64::try_from(count).map_err(|_| invalid("field is too large"))?;
        self.inner.seek(SeekFrom::Current(count))?;
        self.pos += count as u64;
        Ok(())
    }

    fn bytes(&mut self, count: u64) -> io::Result<Vec<u8>> {
        if count > MAX_STRING {
            return Err(invalid("string field is unreasonably large"));
        }
        let mut buffer = vec![0u8; count as usize];
        self.inner.read_exact(&mut buffer)?;
        self.pos += count;
        Ok(buffer)
    }

    fn string(&mut self, count: u64) -> io::Result<String> {
        Ok(String::from_utf8_lossy(&self.bytes(count)?).into_owned())
    }

    /// The next field of a message ending at `end`, or `None`.
    fn field(&mut self, end: u64) -> io::Result<Option<(u32, u8)>> {
        if self.pos >= end {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = u32::try_from(key >> 3).map_err(|_| invalid("bad field number"))?;
        Ok(Some((field, (key & 7) as u8)))
    }

    fn skip_value(&mut self, wire: u8) -> io::Result<()> {
        match wire {
            0 => self.varint().map(|_| ()),
            1 => self.skip(8),
            2 => {
                let len = self.varint()?;
                self.skip(len)
            }
            5 => self.skip(4),
            _ => Err(invalid("unsupported protobuf wire type")),
        }
    }

    /// Reads a length prefix and returns the end of the embedded message.
    fn message_end(&mut self, limit: u64) -> io::Result<u64> {
        let len = self.varint()?;
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| invalid("bad length"))?;
        if end > limit {
            return Err(invalid("embedded message overruns its parent"));
        }
        Ok(end)
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("not a valid ONNX model: {message}"),
    )
}

fn expect_len(wire: u8) -> io::Result<()> {
    if wire == 2 {
        Ok(())
    } else {
        Err(invalid("expected a length-delimited field"))
    }
}

/// `StringStringEntryProto` → (key, value).
fn read_entry<R: Read + Seek>(wire: &mut Wire<R>, end: u64) -> io::Result<(String, String)> {
    let mut key = String::new();
    let mut value = String::new();
    while let Some((field, kind)) = wire.field(end)? {
        match (field, kind) {
            (1, 2) => {
                let len = wire.varint()?;
                key = wire.string(len)?;
            }
            (2, 2) => {
                let len = wire.varint()?;
                value = wire.string(len)?;
            }
            (_, kind) => wire.skip_value(kind)?,
        }
    }
    Ok((key, value))
}

/// `TensorProto` → (name, external location).
fn read_tensor<R: Read + Seek>(
    wire: &mut Wire<R>,
    end: u64,
) -> io::Result<(String, Option<String>)> {
    let mut name = String::new();
    let mut location = None;
    while let Some((field, kind)) = wire.field(end)? {
        match (field, kind) {
            (8, 2) => {
                let len = wire.varint()?;
                name = wire.string(len)?;
            }
            (13, 2) => {
                let entry_end = wire.message_end(end)?;
                let (key, value) = read_entry(wire, entry_end)?;
                if key == "location" {
                    location = Some(value);
                }
            }
            (_, kind) => wire.skip_value(kind)?,
        }
    }
    Ok((name, location))
}

/// `TensorShapeProto.Dimension`.
fn read_dim<R: Read + Seek>(wire: &mut Wire<R>, end: u64) -> io::Result<Dim> {
    let mut dim = Dim::Unknown;
    while let Some((field, kind)) = wire.field(end)? {
        match (field, kind) {
            (1, 0) => dim = Dim::Fixed(wire.varint()? as i64),
            (2, 2) => {
                let len = wire.varint()?;
                dim = Dim::Symbolic(wire.string(len)?);
            }
            (_, kind) => wire.skip_value(kind)?,
        }
    }
    Ok(dim)
}

/// `TypeProto` → (elem type, shape) for tensors.
fn read_type<R: Read + Seek>(
    wire: &mut Wire<R>,
    end: u64,
) -> io::Result<(Option<ElemType>, Option<Vec<Dim>>)> {
    let mut elem = None;
    let mut shape = None;
    while let Some((field, kind)) = wire.field(end)? {
        match (field, kind) {
            // tensor_type
            (1, 2) => {
                let tensor_end = wire.message_end(end)?;
                while let Some((field, kind)) = wire.field(tensor_end)? {
                    match (field, kind) {
                        (1, 0) => elem = Some(ElemType::from_onnx(wire.varint()? as i32)),
                        (2, 2) => {
                            let shape_end = wire.message_end(tensor_end)?;
                            let mut dims = Vec::new();
                            while let Some((field, kind)) = wire.field(shape_end)? {
                                if (field, kind) == (1, 2) {
                                    let dim_end = wire.message_end(shape_end)?;
                                    dims.push(read_dim(wire, dim_end)?);
                                } else {
                                    wire.skip_value(kind)?;
                                }
                            }
                            shape = Some(dims);
                        }
                        (_, kind) => wire.skip_value(kind)?,
                    }
                }
            }
            (_, kind) => wire.skip_value(kind)?,
        }
    }
    Ok((elem, shape))
}

/// `ValueInfoProto`.
fn read_value_info<R: Read + Seek>(wire: &mut Wire<R>, end: u64) -> io::Result<TensorInfo> {
    let mut info = TensorInfo {
        name: String::new(),
        elem_type: None,
        shape: None,
    };
    while let Some((field, kind)) = wire.field(end)? {
        match (field, kind) {
            (1, 2) => {
                let len = wire.varint()?;
                info.name = wire.string(len)?;
            }
            (2, 2) => {
                let type_end = wire.message_end(end)?;
                let (elem, shape) = read_type(wire, type_end)?;
                info.elem_type = elem;
                info.shape = shape;
            }
            (_, kind) => wire.skip_value(kind)?,
        }
    }
    Ok(info)
}

struct Graph {
    inputs: Vec<TensorInfo>,
    outputs: Vec<TensorInfo>,
    initializers: Vec<String>,
    external: Vec<String>,
}

fn read_graph<R: Read + Seek>(wire: &mut Wire<R>, end: u64) -> io::Result<Graph> {
    let mut graph = Graph {
        inputs: Vec::new(),
        outputs: Vec::new(),
        initializers: Vec::new(),
        external: Vec::new(),
    };
    let note_location = |graph: &mut Graph, location: Option<String>| {
        if let Some(location) = location
            && !graph.external.contains(&location)
        {
            graph.external.push(location);
        }
    };
    while let Some((field, kind)) = wire.field(end)? {
        match (field, kind) {
            // initializer
            (5, _) => {
                expect_len(kind)?;
                let tensor_end = wire.message_end(end)?;
                let (name, location) = read_tensor(wire, tensor_end)?;
                graph.initializers.push(name);
                note_location(&mut graph, location);
            }
            // sparse_initializer: values (1) and indices (2) are tensors
            (15, _) => {
                expect_len(kind)?;
                let sparse_end = wire.message_end(end)?;
                while let Some((field, kind)) = wire.field(sparse_end)? {
                    if matches!(field, 1 | 2) && kind == 2 {
                        let tensor_end = wire.message_end(sparse_end)?;
                        let (name, location) = read_tensor(wire, tensor_end)?;
                        if field == 1 {
                            graph.initializers.push(name);
                        }
                        note_location(&mut graph, location);
                    } else {
                        wire.skip_value(kind)?;
                    }
                }
            }
            (11, _) => {
                expect_len(kind)?;
                let info_end = wire.message_end(end)?;
                graph.inputs.push(read_value_info(wire, info_end)?);
            }
            (12, _) => {
                expect_len(kind)?;
                let info_end = wire.message_end(end)?;
                graph.outputs.push(read_value_info(wire, info_end)?);
            }
            (_, kind) => wire.skip_value(kind)?,
        }
    }
    Ok(graph)
}

/// Reads a model summary from any seekable source of `len` bytes.
pub fn scan<R: Read + Seek>(source: R, len: u64) -> io::Result<OnnxSummary> {
    let mut wire = Wire::new(source);
    let mut summary = OnnxSummary::default();
    let mut graph = None;
    while let Some((field, kind)) = wire.field(len)? {
        match (field, kind) {
            (1, 0) => summary.ir_version = wire.varint()? as i64,
            (2, 2) => {
                let count = wire.varint()?;
                summary.producer = wire.string(count)?;
            }
            (7, 2) => {
                let end = wire.message_end(len)?;
                graph = Some(read_graph(&mut wire, end)?);
            }
            (8, 2) => {
                let end = wire.message_end(len)?;
                let mut domain = String::new();
                let mut version = 0;
                while let Some((field, kind)) = wire.field(end)? {
                    match (field, kind) {
                        (1, 2) => {
                            let count = wire.varint()?;
                            domain = wire.string(count)?;
                        }
                        (2, 0) => version = wire.varint()? as i64,
                        (_, kind) => wire.skip_value(kind)?,
                    }
                }
                summary.opsets.push((domain, version));
            }
            (14, 2) => {
                let end = wire.message_end(len)?;
                let (key, value) = read_entry(&mut wire, end)?;
                summary.metadata.insert(key, value);
            }
            (_, kind) => wire.skip_value(kind)?,
        }
    }
    let graph = graph.ok_or_else(|| invalid("the model has no graph"))?;
    summary.inputs = graph
        .inputs
        .into_iter()
        .filter(|input| !graph.initializers.contains(&input.name))
        .collect();
    summary.outputs = graph.outputs;
    summary.external_data = graph.external;
    Ok(summary)
}

/// Reads a model summary from a file.
pub fn scan_file(path: &Path) -> io::Result<OnnxSummary> {
    let file = File::open(path)?;
    let len = file.metadata()?.len();
    scan(BufReader::with_capacity(64 * 1024, file), len)
}

/// Whether an external-data location stays inside the model's directory.
pub fn is_safe_location(location: &str) -> bool {
    let path = Path::new(location);
    !location.is_empty()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

#[cfg(test)]
pub(crate) mod encode {
    //! A tiny protobuf encoder for building test models.

    pub fn varint(mut value: u64, out: &mut Vec<u8>) {
        loop {
            let byte = (value & 0x7F) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    pub fn field_varint(field: u32, value: u64, out: &mut Vec<u8>) {
        varint(u64::from(field) << 3, out);
        varint(value, out);
    }

    pub fn field_bytes(field: u32, bytes: &[u8], out: &mut Vec<u8>) {
        varint((u64::from(field) << 3) | 2, out);
        varint(bytes.len() as u64, out);
        out.extend_from_slice(bytes);
    }

    pub fn entry(key: &str, value: &str) -> Vec<u8> {
        let mut out = Vec::new();
        field_bytes(1, key.as_bytes(), &mut out);
        field_bytes(2, value.as_bytes(), &mut out);
        out
    }

    /// `ValueInfoProto` for a tensor; negative dims are symbolic.
    pub fn value_info(name: &str, elem: i32, dims: &[i64]) -> Vec<u8> {
        let mut shape = Vec::new();
        for dim in dims {
            let mut d = Vec::new();
            if *dim >= 0 {
                field_varint(1, *dim as u64, &mut d);
            } else {
                field_bytes(2, b"batch", &mut d);
            }
            field_bytes(1, &d, &mut shape);
        }
        let mut tensor = Vec::new();
        field_varint(1, elem as u64, &mut tensor);
        field_bytes(2, &shape, &mut tensor);
        let mut ty = Vec::new();
        field_bytes(1, &tensor, &mut ty);
        let mut info = Vec::new();
        field_bytes(1, name.as_bytes(), &mut info);
        field_bytes(2, &ty, &mut info);
        info
    }

    /// An initializer tensor, either external or with `raw` bytes inline.
    pub fn initializer(name: &str, external: Option<&str>, raw: usize) -> Vec<u8> {
        let mut tensor = Vec::new();
        field_varint(1, raw as u64, &mut tensor);
        field_varint(2, 1, &mut tensor);
        field_bytes(8, name.as_bytes(), &mut tensor);
        match external {
            Some(location) => {
                field_bytes(13, &entry("location", location), &mut tensor);
                field_bytes(13, &entry("offset", "0"), &mut tensor);
                field_varint(14, 1, &mut tensor);
            }
            None => field_bytes(9, &vec![7u8; raw], &mut tensor),
        }
        tensor
    }

    pub struct ModelSpec<'a> {
        pub inputs: Vec<(&'a str, i32, Vec<i64>)>,
        pub outputs: Vec<(&'a str, i32, Vec<i64>)>,
        pub initializers: Vec<(&'a str, Option<&'a str>, usize)>,
        pub metadata: Vec<(&'a str, &'a str)>,
        /// Also list initializers as graph inputs (old IR versions do).
        pub initializers_as_inputs: bool,
    }

    pub fn model(spec: &ModelSpec) -> Vec<u8> {
        let mut graph = Vec::new();
        // A node, to be skipped.
        field_bytes(1, b"\x0a\x01x\x12\x01y\x22\x04Relu", &mut graph);
        for (name, external, raw) in &spec.initializers {
            field_bytes(5, &initializer(name, *external, *raw), &mut graph);
            if spec.initializers_as_inputs {
                field_bytes(11, &value_info(name, 1, &[*raw as i64]), &mut graph);
            }
        }
        for (name, elem, dims) in &spec.inputs {
            field_bytes(11, &value_info(name, *elem, dims), &mut graph);
        }
        for (name, elem, dims) in &spec.outputs {
            field_bytes(12, &value_info(name, *elem, dims), &mut graph);
        }
        let mut model = Vec::new();
        field_varint(1, 10, &mut model);
        field_bytes(2, b"test", &mut model);
        let mut opset = Vec::new();
        field_bytes(1, b"", &mut opset);
        field_varint(2, 18, &mut opset);
        field_bytes(8, &opset, &mut model);
        field_bytes(7, &graph, &mut model);
        for (key, value) in &spec.metadata {
            field_bytes(14, &entry(key, value), &mut model);
        }
        model
    }
}

#[cfg(test)]
mod tests {
    use super::encode::*;
    use super::*;
    use std::io::Cursor;

    fn scan_bytes(bytes: Vec<u8>) -> io::Result<OnnxSummary> {
        let len = bytes.len() as u64;
        scan(Cursor::new(bytes), len)
    }

    #[test]
    fn reads_inputs_outputs_metadata_and_external_data() {
        let bytes = model(&ModelSpec {
            inputs: vec![("input", 1, vec![32, 3, 256, 256])],
            outputs: vec![
                ("reconstruction", 1, vec![32, 3, 256, 256]),
                ("mu", 10, vec![-1, 64]),
            ],
            initializers: vec![
                ("w1", Some("model.onnx.data"), 0),
                ("w2", Some("model.onnx.data"), 0),
                ("w3", None, 100_000),
            ],
            metadata: vec![("eovae.mpp", "0.25")],
            initializers_as_inputs: true,
        });
        let summary = scan_bytes(bytes).unwrap();
        assert_eq!(summary.ir_version, 10);
        assert_eq!(summary.producer, "test");
        assert_eq!(summary.opsets, vec![(String::new(), 18)]);
        assert_eq!(summary.metadata["eovae.mpp"], "0.25");
        // Initializers listed as inputs are not inputs to feed.
        assert_eq!(summary.inputs.len(), 1);
        let input = &summary.inputs[0];
        assert_eq!(input.name, "input");
        assert_eq!(input.elem_type, Some(ElemType::Float));
        assert_eq!(input.shape_label(), "[32 × 3 × 256 × 256]");
        assert_eq!(summary.outputs[1].elem_type, Some(ElemType::Float16));
        assert_eq!(
            summary.outputs[1].shape.as_ref().unwrap()[0],
            Dim::Symbolic("batch".into())
        );
        assert_eq!(summary.external_data, vec!["model.onnx.data".to_string()]);
    }

    #[test]
    fn rejects_garbage_and_truncation() {
        assert!(scan_bytes(b"not a model at all".to_vec()).is_err());
        let mut bytes = model(&ModelSpec {
            inputs: vec![("input", 1, vec![1, 3, 8, 8])],
            outputs: vec![("output", 1, vec![1, 3, 8, 8])],
            initializers: vec![],
            metadata: vec![],
            initializers_as_inputs: false,
        });
        bytes.truncate(bytes.len() - 5);
        assert!(scan_bytes(bytes).is_err());
        assert!(scan_bytes(Vec::new()).is_err(), "no graph");
    }

    #[test]
    fn locations_must_stay_inside_the_model_directory() {
        assert!(is_safe_location("model.onnx.data"));
        assert!(is_safe_location("weights/part1.bin"));
        assert!(!is_safe_location("../secret"));
        assert!(!is_safe_location("/etc/passwd"));
        assert!(!is_safe_location(""));
    }

    #[test]
    fn reads_the_reference_model_when_present() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("HistoVAE/pretrained/HistoVae_reconstruct.onnx");
        if !path.exists() {
            return;
        }
        let summary = scan_file(&path).unwrap();
        assert_eq!(summary.inputs.len(), 1);
        assert_eq!(summary.inputs[0].name, "input");
        assert_eq!(summary.inputs[0].shape_label(), "[32 × 3 × 256 × 256]");
        assert_eq!(summary.outputs[0].name, "reconstruction");
        assert_eq!(
            summary.external_data,
            vec!["HistoVae_reconstruct.onnx.data".to_string()]
        );
        assert!(summary.metadata.is_empty());
    }
}
