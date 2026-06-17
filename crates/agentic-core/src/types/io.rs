use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

use crate::vector_search::types::SearchOptions;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputTextContent {
    #[serde(rename = "type")]
    pub type_: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputImageContent {
    #[serde(rename = "type")]
    pub type_: String,
    pub image_url: Option<String>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputContent {
    #[serde(rename = "input_text")]
    Text(InputTextContent),
    #[serde(rename = "input_image")]
    Image(InputImageContent),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMessage {
    pub role: String,
    pub content: InputMessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputMessageContent {
    Text(String),
    Parts(Vec<InputContent>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionToolResultMessage {
    pub call_id: String,
    pub output: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum InputItem {
    #[serde(rename = "message")]
    Message(InputMessage),
    #[serde(rename = "function_call")]
    FunctionCall(FunctionToolCall),
    #[serde(rename = "function_call_output")]
    FunctionCallOutput(FunctionToolResultMessage),
    #[serde(other)]
    Unknown,
}

impl<'de> Deserialize<'de> for InputItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut value = Value::deserialize(deserializer)?;
        let Some(object) = value.as_object_mut() else {
            return Ok(Self::Unknown);
        };

        match object.get("type").and_then(Value::as_str) {
            Some("message") => {
                object.remove("type");
                serde_json::from_value(value)
                    .map(Self::Message)
                    .map_err(serde::de::Error::custom)
            }
            Some("function_call") => {
                object.remove("type");
                serde_json::from_value(value)
                    .map(Self::FunctionCall)
                    .map_err(serde::de::Error::custom)
            }
            Some("function_call_output") => {
                object.remove("type");
                serde_json::from_value(value)
                    .map(Self::FunctionCallOutput)
                    .map_err(serde::de::Error::custom)
            }
            None if object.contains_key("role") && object.contains_key("content") => serde_json::from_value(value)
                .map(Self::Message)
                .map_err(serde::de::Error::custom),
            Some(_) | None => Ok(Self::Unknown),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTextContent {
    #[serde(rename = "type")]
    pub type_: String,
    pub text: String,
    #[serde(default)]
    pub annotations: Vec<Value>,
}

impl OutputTextContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            type_: "output_text".into(),
            text: text.into(),
            annotations: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputMessage {
    #[serde(default)]
    pub id: String,
    pub role: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub content: Vec<OutputTextContent>,
}

impl OutputMessage {
    pub fn new(id: impl Into<String>, status: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role: "assistant".into(),
            status: status.into(),
            content: vec![],
        }
    }
}

impl From<OutputMessage> for InputMessage {
    fn from(msg: OutputMessage) -> Self {
        let parts = msg
            .content
            .into_iter()
            .map(|c| {
                InputContent::Text(InputTextContent {
                    type_: c.type_,
                    text: c.text,
                })
            })
            .collect();
        Self {
            role: msg.role,
            content: InputMessageContent::Parts(parts),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionToolCall {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    #[serde(rename = "message")]
    Message(OutputMessage),
    #[serde(rename = "function_call")]
    FunctionCall(FunctionToolCall),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct InputTokenDetails {
    pub cached_tokens: i64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct OutputTokenDetails {
    pub reasoning_tokens: i64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub total_tokens: i64,
    #[serde(default)]
    pub input_tokens_details: InputTokenDetails,
    #[serde(default)]
    pub output_tokens_details: OutputTokenDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSearchTool {
    #[serde(default)]
    pub vector_store_ids: Vec<String>,
    #[serde(default, flatten)]
    pub search_options: SearchOptions,
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesTool {
    #[serde(rename = "function")]
    Function(FunctionTool),
    #[serde(rename = "file_search")]
    FileSearch(FileSearchTool),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    #[serde(rename = "function")]
    Function {
        name: String,
    },
}

/// Returns the effective tool list, preferring `request_tools` when explicitly
/// set by the caller, otherwise falling back to the stored configuration.
#[inline]
pub(crate) fn resolve_tools(
    request_tools: Option<&[ResponsesTool]>,
    stored_tools: Option<&[ResponsesTool]>,
    tools_explicitly_set: bool,
) -> Option<Vec<ResponsesTool>> {
    if tools_explicitly_set {
        request_tools
    } else {
        stored_tools
    }
    .map(<[_]>::to_vec)
}

/// Returns the effective tool choice using the same precedence as [`resolve_tools`].
#[inline]
pub(crate) fn resolve_tool_choice(
    request_choice: &ToolChoice,
    stored_choice: &ToolChoice,
    explicitly_set: bool,
) -> ToolChoice {
    if explicitly_set { request_choice } else { stored_choice }.clone()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}
