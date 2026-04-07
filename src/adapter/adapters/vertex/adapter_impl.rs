use crate::adapter::adapters::support::get_api_key;
use crate::adapter::anthropic::AnthropicAdapter;
use crate::adapter::gemini::GeminiAdapter;
use crate::adapter::{Adapter, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{ChatOptionsSet, ChatRequest, ChatResponse, ChatStreamResponse};
use crate::resolver::{AuthData, Endpoint};
use crate::webc::WebResponse;
use crate::{Error, Headers, ModelIden, Result, ServiceTarget};
use reqwest::RequestBuilder;
use tracing::warn;
use value_ext::JsonValueExt;

pub struct VertexAdapter;

const VERTEX_ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

impl VertexAdapter {
	pub const API_KEY_DEFAULT_ENV_NAME: &str = "VERTEX_API_KEY";
}

// region:    --- VertexPublisher

/// Internal enum to dispatch wire format based on the model's publisher.
enum VertexPublisher {
	Google,
	Anthropic,
}

impl VertexPublisher {
	fn from_model_name(model_name: &str) -> Result<Self> {
		if model_name.starts_with("gemini") {
			Ok(Self::Google)
		} else if model_name.starts_with("claude") {
			Ok(Self::Anthropic)
		} else {
			Err(Error::AdapterNotSupported {
				adapter_kind: AdapterKind::Vertex,
				feature: format!("model '{model_name}' (unknown Vertex AI publisher)"),
			})
		}
	}

	fn publisher_path(&self) -> &'static str {
		match self {
			Self::Google => "publishers/google",
			Self::Anthropic => "publishers/anthropic",
		}
	}
}

// endregion: --- VertexPublisher

impl Adapter for VertexAdapter {
	const DEFAULT_API_KEY_ENV_NAME: Option<&'static str> = Some(Self::API_KEY_DEFAULT_ENV_NAME);

	fn default_endpoint() -> Endpoint {
		let project_id = std::env::var("VERTEX_PROJECT_ID").unwrap_or_else(|_| {
			warn!("VERTEX_PROJECT_ID env var is not set; Vertex AI requests will use a malformed URL");
			String::new()
		});
		// Model availability varies by region. See https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/locations for details.
		let base_url = match std::env::var("VERTEX_LOCATION") {
			Ok(location) => {
				format!("https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/")
			}
			// When no location is set, fall back to "global"
			Err(_) => format!("https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/"),
		};
		Endpoint::from_owned(base_url)
	}
	fn default_auth() -> AuthData {
		match Self::DEFAULT_API_KEY_ENV_NAME {
			Some(env_name) => AuthData::from_env(env_name),
			None => AuthData::None,
		}
	}

	async fn all_model_names(_kind: AdapterKind, _endpoint: Endpoint, _auth: AuthData) -> Result<Vec<String>> {
		Ok(vec![
			"gemini-2.5-pro".to_string(),
			"gemini-2.5-flash".to_string(),
			"gemini-2.5-flash-lite".to_string(),
			"claude-sonnet-4-6".to_string(),
			"claude-opus-4-6".to_string(),
			"claude-haiku-4-5".to_string(),
		])
	}

	/// Note: An unrecognized model prefix falls back to `VertexPublisher::Google` with a warning.
	/// In practice, `to_web_request_data` validates the publisher first and will return
	/// an error before this fallback is ever reached.
	fn get_service_url(model: &ModelIden, service_type: ServiceType, endpoint: Endpoint) -> Result<String> {
		let base_url = endpoint.base_url();
		let (_, model_name) = model.model_name.namespace_and_name();
		let publisher = VertexPublisher::from_model_name(model_name).unwrap_or_else(|_| {
			warn!("Unknown Vertex AI publisher for model '{model_name}'; falling back to Google publisher");
			VertexPublisher::Google
		});
		let publisher_path = publisher.publisher_path();

		let url = match publisher {
			VertexPublisher::Google => match service_type {
				ServiceType::Chat => format!("{base_url}{publisher_path}/models/{model_name}:generateContent"),
				ServiceType::ChatStream => {
					format!("{base_url}{publisher_path}/models/{model_name}:streamGenerateContent")
				}
				ServiceType::Embed => format!("{base_url}{publisher_path}/models/{model_name}:predict"),
			},
			VertexPublisher::Anthropic => match service_type {
				ServiceType::Chat | ServiceType::ChatStream => {
					format!("{base_url}{publisher_path}/models/{model_name}:rawPredict")
				}
				ServiceType::Embed => format!("{base_url}{publisher_path}/models/{model_name}:predict"),
			},
		};

		Ok(url)
	}

	fn to_web_request_data(
		target: ServiceTarget,
		service_type: ServiceType,
		chat_req: ChatRequest,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		let ServiceTarget { endpoint, auth, model } = target;
		let (_, model_name) = model.model_name.namespace_and_name();
		let publisher = VertexPublisher::from_model_name(model_name)?;
		let model_name = model_name.to_string();

		// For Vertex AI the "api key" is an OAuth2 Bearer token supplied by the AuthResolver
		let api_key = get_api_key(auth, &model)?;
		let headers = Headers::from(("Authorization".to_string(), format!("Bearer {api_key}")));

		match publisher {
			VertexPublisher::Google => Self::to_gemini_web_request_data(
				model,
				&model_name,
				endpoint,
				headers,
				service_type,
				chat_req,
				options_set,
			),
			VertexPublisher::Anthropic => Self::to_anthropic_web_request_data(
				model,
				&model_name,
				endpoint,
				headers,
				service_type,
				chat_req,
				options_set,
			),
		}
	}

	fn to_chat_response(
		model_iden: ModelIden,
		web_response: WebResponse,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatResponse> {
		let (_, model_name) = model_iden.model_name.namespace_and_name();
		let publisher = VertexPublisher::from_model_name(model_name)?;

		match publisher {
			VertexPublisher::Google => GeminiAdapter::to_chat_response(model_iden, web_response, options_set),
			VertexPublisher::Anthropic => AnthropicAdapter::to_chat_response(model_iden, web_response, options_set),
		}
	}

	fn to_chat_stream(
		model_iden: ModelIden,
		reqwest_builder: RequestBuilder,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatStreamResponse> {
		let (_, model_name) = model_iden.model_name.namespace_and_name();
		let publisher = VertexPublisher::from_model_name(model_name)?;

		match publisher {
			VertexPublisher::Google => GeminiAdapter::to_chat_stream(model_iden, reqwest_builder, options_set),
			VertexPublisher::Anthropic => AnthropicAdapter::to_chat_stream(model_iden, reqwest_builder, options_set),
		}
	}

	fn to_embed_request_data(
		_service_target: ServiceTarget,
		_embed_req: crate::embed::EmbedRequest,
		_options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		Err(Error::AdapterNotSupported {
			adapter_kind: AdapterKind::Vertex,
			feature: "embeddings".to_string(),
		})
	}

	fn to_embed_response(
		_model_iden: ModelIden,
		_web_response: WebResponse,
		_options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<crate::embed::EmbedResponse> {
		Err(Error::AdapterNotSupported {
			adapter_kind: AdapterKind::Vertex,
			feature: "embeddings".to_string(),
		})
	}
}

// region:    --- Gemini Publisher Support

impl VertexAdapter {
	fn to_gemini_web_request_data(
		model: ModelIden,
		model_name: &str,
		endpoint: Endpoint,
		headers: Headers,
		service_type: ServiceType,
		chat_req: ChatRequest,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		let (payload, provider_model_name) =
			GeminiAdapter::build_gemini_request_payload(&model, model_name, chat_req, options_set)?;

		let provider_model = model.from_name(&provider_model_name);
		let url = Self::get_service_url(&provider_model, service_type, endpoint)?;

		Ok(WebRequestData { url, headers, payload })
	}
}

// endregion: --- Gemini Publisher Support

// region:    --- Anthropic Publisher Support

impl VertexAdapter {
	fn to_anthropic_web_request_data(
		model: ModelIden,
		model_name: &str,
		endpoint: Endpoint,
		headers: Headers,
		service_type: ServiceType,
		chat_req: ChatRequest,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		// Reuse the shared Anthropic payload builder (mirrors the Gemini pattern above).
		// Vertex Claude: model is in URL, not body; anthropic_version goes in body instead of "model".
		let (mut payload, resolved_model_name) =
			AnthropicAdapter::build_anthropic_request_payload(model_name, service_type, chat_req, &options_set)?;
		payload.x_insert("anthropic_version", VERTEX_ANTHROPIC_VERSION)?;

		// Vertex Claude does not support structured output via output_config.format
		// (only the direct Anthropic API and Amazon Bedrock do).
		// Strip format from output_config so the request isn't rejected, and warn the caller.
		// See: https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs
		if let Some(output_config) = payload.get_mut("output_config").and_then(|v| v.as_object_mut()) {
			if output_config.remove("format").is_some() {
				warn!(
					"Vertex Claude does not support structured output (output_config.format). \
					 The response_format option has been ignored for model '{model_name}'. \
					 Use the direct Anthropic adapter or Amazon Bedrock for constrained JSON output."
				);
			}
			// If output_config is now empty (no effort either), remove it entirely
			if output_config.is_empty() {
				payload.as_object_mut().map(|m| m.remove("output_config"));
			}
		}

		// Use the resolved (suffix-stripped) model name for the URL, just like the Gemini path,
		// so reasoning suffixes like "-high" don't leak into the Vertex AI endpoint URL.
		let provider_model = model.from_name(&resolved_model_name);
		let url = Self::get_service_url(&provider_model, service_type, endpoint)?;

		Ok(WebRequestData { url, headers, payload })
	}
}

// endregion: --- Anthropic Publisher Support

// region:    --- Tests

#[cfg(test)]
mod tests {
	use super::*;
	use crate::adapter::ServiceType;
	use crate::chat::{ChatOptions, ChatOptionsSet, ChatRequest, ChatResponseFormat, JsonSpec};
	use serde_json::json;

	/// Verifies that the Vertex Claude path includes anthropic_version, does NOT include
	/// "model" (model is in the URL), and strips output_config.format since Vertex
	/// Claude doesn't support structured output.
	#[test]
	fn test_vertex_claude_payload_has_anthropic_version_and_no_model() {
		let options_set = ChatOptionsSet::default();

		let model_iden = ModelIden::new(AdapterKind::Vertex, "claude-sonnet-4-6");
		let endpoint = VertexAdapter::default_endpoint();

		let web_req = VertexAdapter::to_anthropic_web_request_data(
			model_iden,
			"claude-sonnet-4-6",
			endpoint,
			Headers::default(),
			ServiceType::Chat,
			ChatRequest::from_user("hello"),
			options_set,
		)
		.expect("to_anthropic_web_request_data should succeed");

		let payload = &web_req.payload;

		// anthropic_version must be present (Vertex Claude discriminator)
		assert_eq!(
			payload.get("anthropic_version").and_then(|v| v.as_str()),
			Some(VERTEX_ANTHROPIC_VERSION),
			"anthropic_version must be set to the Vertex version string"
		);

		// "model" must NOT be in the payload -- Vertex puts model in URL
		assert!(payload.get("model").is_none(), "model must not be in Vertex Claude payload");
	}

	/// Verifies that output_config.format (structured output) is stripped from the
	/// Vertex Claude payload because Vertex doesn't support it. If only format was
	/// in output_config, the entire key should be removed.
	#[test]
	fn test_vertex_claude_strips_unsupported_output_config_format() {
		let chat_options = ChatOptions {
			response_format: Some(ChatResponseFormat::JsonSpec(JsonSpec::new(
				"test_schema",
				json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
			))),
			..Default::default()
		};
		let options_set = ChatOptionsSet::default().with_chat_options(Some(&chat_options));

		let model_iden = ModelIden::new(AdapterKind::Vertex, "claude-sonnet-4-6");
		let endpoint = VertexAdapter::default_endpoint();

		let web_req = VertexAdapter::to_anthropic_web_request_data(
			model_iden,
			"claude-sonnet-4-6",
			endpoint,
			Headers::default(),
			ServiceType::Chat,
			ChatRequest::from_user("hello"),
			options_set,
		)
		.expect("to_anthropic_web_request_data should succeed");

		let payload = &web_req.payload;

		// output_config.format must be stripped (Vertex Claude doesn't support structured output)
		assert!(
			payload.get("output_config").is_none(),
			"output_config should be removed entirely when only format was present"
		);
	}

	/// Verifies that when both reasoning effort and structured output are set, the
	/// Vertex path preserves output_config.effort but strips output_config.format.
	#[test]
	fn test_vertex_claude_keeps_effort_but_strips_format() {
		use crate::chat::ReasoningEffort;

		let chat_options = ChatOptions {
			reasoning_effort: Some(ReasoningEffort::High),
			response_format: Some(ChatResponseFormat::JsonSpec(JsonSpec::new(
				"test_schema",
				json!({"type": "object", "properties": {}}),
			))),
			..Default::default()
		};
		let options_set = ChatOptionsSet::default().with_chat_options(Some(&chat_options));

		let model_iden = ModelIden::new(AdapterKind::Vertex, "claude-sonnet-4-6");
		let endpoint = VertexAdapter::default_endpoint();

		let web_req = VertexAdapter::to_anthropic_web_request_data(
			model_iden,
			"claude-sonnet-4-6",
			endpoint,
			Headers::default(),
			ServiceType::Chat,
			ChatRequest::from_user("hello"),
			options_set,
		)
		.expect("to_anthropic_web_request_data should succeed");

		let payload = &web_req.payload;
		let output_config = payload.get("output_config").expect("output_config must be present (effort remains)");

		// effort should be preserved
		assert_eq!(
			output_config.get("effort").and_then(|v| v.as_str()),
			Some("high"),
			"effort must be preserved in output_config"
		);

		// format must be stripped
		assert!(
			output_config.get("format").is_none(),
			"format must be stripped from output_config on Vertex Claude"
		);
	}
}

// endregion: --- Tests
